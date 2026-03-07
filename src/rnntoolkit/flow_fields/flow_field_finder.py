import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from rnntoolkit.linear import Linearization
from rnntoolkit.flow_fields.flow_field import FlowField
from rnntoolkit.flow_fields.flow_field_finder_base import FlowFieldFinderBase


class FlowFieldFinder(FlowFieldFinderBase):
    def __init__(
        self,
        rnn: nn.RNN,
        num_points: int,
        x_offset: int,
        y_offset: int,
        x_center: int,
        y_center: int,
        follow_traj: bool = False,
    ):
        super().__init__(rnn, num_points, x_offset, y_offset, x_center, y_center)
        """
        Flow field that gathers a flow field about a specified trajectory

        Args:
            mrnn (RNN): RNN-like object
            num_points (int): number of points to use in grid, results in (num_points, num_points)
            x_offset (int): scale to offset grid about trajectory in x direction
            y_offset (int): scale to offset grid about trajectory in y direction
            cancel_other_regions (bool): whether or not to zero out activity from other regions
            follow_traj (bool): whether or not to center the grid around each trajectory
        """
        self.follow_traj = follow_traj

        # class objects
        self.reduce_obj = PCA(n_components=2)
        self.linearization = Linearization(self.rnn)

    def find_nonlinear_flow(
        self, states: torch.Tensor, inp: torch.Tensor, traj_to_reduce: torch.Tensor
    ) -> list:
        """Compute 2D flow fields in a region subspace along a trajectory.

        Projects selected region activity onto a 2D PCA subspace, constructs a grid
        around the current point, and advances the system by one step to estimate
        the local flow (velocity vectors). Can zero out non-selected regions or
        keep their control values.

        Args:
            states (torch.Tensor): Hidden activations over time [batch_size, T, N].
            inp (torch.Tensor): External input sequence.
            stim_input (torch.Tensor | None): Optional additive stimulus input.
            W (torch.Tensor | None): Optional weight matrix to use.

        Returns:
            list: FlowField object per sampled time.
        """

        flow_field_list = []

        # Reshape to nxd
        states, inp = self._nxd(states), self._nxd(inp)

        # assert states and input match shape
        assert states.shape[0] == inp.shape[0]
        n_states = states.shape[0]

        self._fit_traj(traj_to_reduce)
        reduced_traj = self._reduce_traj(states)

        # Now going through trajectory
        for n in range(n_states):
            reduced_traj_n, inp_n = reduced_traj[n], inp[n]
            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_tv_bounds(reduced_traj_n)
                )
            else:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_bounds()
                )

            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Repeat along the batch dimension to match the grid
            full_inp_batch = inp_n.repeat(low_dim_grid.shape[0], 1)

            with torch.no_grad():
                # Current timestep input
                # Get activity for current timestep
                _, h = self.rnn(
                    full_inp_batch.unsqueeze(self.time_dim),
                    inverse_grid.unsqueeze(0),
                )

            # Reduce h_next
            h_next = self._reduce_traj(h)

            # Compute velocity and speed
            x_vel, y_vel = self._compute_velocity(h_next, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            # Reshape to match FlowField object requirements
            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )
            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)
            flow_field_list.append(flow_field)

        return flow_field_list

    def find_linear_flow(
        self,
        states: torch.Tensor,
        inp: torch.Tensor,
        delta_inp: torch.Tensor,
        traj_to_reduce: torch.Tensor,
    ) -> list:
        """Compute linearized flow fields in a 2D subspace.

        Similar to :func:`flow_field`, but uses a local linear approximation (Jacobian)
        of the dynamics around points on the trajectory instead of a full forward
        step. Assumes no external input to the selected regions.

        Args:
            states (torch.Tensor): Hidden activations over time for selected regions, [n, d]
        Returns:
            list: FlowField objects per sampled time.
        """

        # reshape to nxd
        states, inp, delta_inp = self._nxd(states), self._nxd(inp), self._nxd(delta_inp)

        assert states.shape[0] == delta_inp.shape[0]
        n_states = states.shape[0]

        # Lists for x and y velocities
        flow_field_list = []

        # Reduce the regional trajectories and return pca object
        self._fit_traj(traj_to_reduce)
        reduced_traj = self._reduce_traj(states)

        for n in range(n_states):
            states_n = states[n]
            reduced_traj_n = reduced_traj[n]
            inp_n = inp[n]
            delta_inp_n = delta_inp[n]

            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_tv_bounds(reduced_traj_n)
                )
            else:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_bounds()
                )

            # Inverse the grid to pass through RNN
            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Get a perturbation of the activity
            delta_h = inverse_grid - states_n

            with torch.no_grad():
                # call forward method for linearization to get affine transformation
                h = self.linearization(inp_n, states_n, delta_inp_n, delta_h)

            # Put next h into a grid format
            h_next = self._reduce_traj(h)

            # Compute velocities between gathered trajectory of grid and original grid values
            x_vel, y_vel = self._compute_velocity(h_next, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            # Reshape data back to grid
            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)
            flow_field_list.append(flow_field)

        return flow_field_list
