import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from dsatorch.linear import Linearization
from dsatorch.flow_fields.flow_field import FlowField
from typing import Generic, TypeVar

RNN = TypeVar("RNN", bound=nn.Module)


class FlowFieldFinder(Generic[RNN]):
    _default_hps = {
        "num_components": 2,
        "num_points": 50,
        "x_offset": 1,
        "y_offset": 1,
        "cancel_other_regions": False,
        "follow_traj": False,
        "name": "run",
        "dtype": torch.float32,
    }

    def __init__(
        self,
        rnn: RNN,
        num_points: int = _default_hps["num_points"],
        x_offset: int = _default_hps["x_offset"],
        y_offset: int = _default_hps["y_offset"],
        cancel_other_regions: bool = _default_hps["cancel_other_regions"],
        follow_traj: bool = _default_hps["follow_traj"],
        dtype=_default_hps["dtype"],
    ):
        """
        Flow field that gathers a flow field about a specified trajectory

        Args:
            mrnn (mRNN): mRNN object
            num_points (int): number of points to use in grid, results in (num_points, num_points)
            x_offset (int): scale to offset grid about trajectory in x direction
            y_offset (int): scale to offset grid about trajectory in y direction
            cancel_other_regions (bool): whether or not to zero out activity from other regions
            follow_traj (bool): whether or not to center the grid around each trajectory
        """
        # Hyperparameters
        self.rnn = rnn
        self.num_points = num_points
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.cancel_other_regions = cancel_other_regions
        self.follow_traj = follow_traj
        self.dtype = dtype

        # class objects
        self.reduce_obj = PCA(n_components=2)
        self.linearization = Linearization(self.rnn)

    def find_nonlinear_flow(
        self, states: torch.Tensor, inp: torch.Tensor, **kwargs
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

        verbose = kwargs["verbose"] if "verbose" in kwargs else False

        flow_field_list = []

        if self.rnn.batch_first:
            time_dim = 1
        else:
            time_dim = 0

        if states.dim() == 1:
            states = states.unsqueeze(0)

        if inp.dim() == 1:
            inp = inp.unsqueeze(0)

        states = torch.flatten(states, end_dim=-2)
        inp = torch.flatten(inp, end_dim=-2)

        assert states.shape[0] == inp.shape[0]
        n_states = states.shape[0]

        reduced_traj = self._reduce_traj(states)

        lower_bound_x = -self.x_offset
        upper_bound_x = self.x_offset
        lower_bound_y = -self.y_offset
        upper_bound_y = self.y_offset

        max_x_vels, max_y_vels, max_speeds = [], [], []
        min_x_vels, min_y_vels, min_speeds = [], [], []

        # Now going through trajectory
        for n in range(1, n_states):
            if self.follow_traj:
                lower_bound_x = torch.round(
                    reduced_traj[n, 0] - self.x_offset, decimals=1
                ).item()
                upper_bound_x = torch.round(
                    reduced_traj[n, 0] + self.x_offset, decimals=1
                ).item()
                lower_bound_y = torch.round(
                    reduced_traj[n, 1] - self.y_offset, decimals=1
                ).item()
                upper_bound_y = torch.round(
                    reduced_traj[n, 1] + self.y_offset, decimals=1
                ).item()

            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Repeat along the batch dimension to match the grid
            full_inp_batch = inp[n].repeat(low_dim_grid.shape[0], 1)

            with torch.no_grad():
                # Current timestep input
                # Get activity for current timestep
                _, h = self.rnn(
                    full_inp_batch.unsqueeze(time_dim),
                    inverse_grid.unsqueeze(0),
                )

            # Get activity for regions of interest
            cur_region_h = torch.reshape(h, (-1, h.shape[-1]))
            cur_region_h = self.reduce_obj.transform(cur_region_h)
            cur_region_h = torch.tensor(cur_region_h, dtype=self.dtype)

            x_vel, y_vel = self._compute_velocity(cur_region_h, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            # Reshape to match FlowField object requirements
            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)
            flow_field_list.append(flow_field)

            # append max values to lists
            max_x_vels.append(flow_field.max_x_vel)
            max_y_vels.append(flow_field.max_y_vel)
            max_speeds.append(flow_field.max_speed)

            # append min values to lists
            min_x_vels.append(flow_field.min_x_vel)
            min_y_vels.append(flow_field.min_y_vel)
            min_speeds.append(flow_field.min_speed)

        if verbose:
            mean_max_x_vel, std_max_x_vel = np.mean(max_x_vels), np.std(max_x_vels)
            mean_max_y_vel, std_max_y_vel = np.mean(max_y_vels), np.std(max_y_vels)
            mean_max_speed, std_max_speed = np.mean(max_speeds), np.std(max_speeds)

            mean_min_x_vel, std_min_x_vel = np.mean(min_x_vels), np.std(min_x_vels)
            mean_min_y_vel, std_min_y_vel = np.mean(min_x_vels), np.std(min_y_vels)
            mean_min_speed, std_min_speed = np.mean(min_x_vels), np.std(min_speeds)

            print("======================")
            print("Flow Field Statistics:")
            print(
                f"mean max x vel: {mean_max_x_vel} +/- {std_max_x_vel}   mean min x vel: {mean_min_x_vel} +/- {std_min_x_vel}"
            )
            print(
                f"mean max y vel: {mean_max_y_vel} +/- {std_max_y_vel}   mean min y vel: {mean_min_y_vel} +/- {std_min_y_vel}"
            )
            print(
                f"mean max speed: {mean_max_speed} +/- {std_max_speed}   mean min speed: {mean_min_speed} +/- {std_min_speed}"
            )
            print("======================")

        return flow_field_list

    def find_linear_flow(
        self,
        states: torch.Tensor,
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
        # Lists for x and y velocities
        flow_field_list = []

        states = torch.flatten(states, end_dim=-2)
        n_states = states.shape[0]

        # Reduce the regional trajectories and return pca object
        reduced_traj = self._reduce_traj(states)

        # Grid offsets
        lower_bound_x = -self.x_offset
        upper_bound_x = self.x_offset
        lower_bound_y = -self.y_offset
        upper_bound_y = self.y_offset

        for n in range(1, n_states):
            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x = torch.round(
                    reduced_traj[n, 0] - self.x_offset, decimals=1
                ).item()
                upper_bound_x = torch.round(
                    reduced_traj[n, 0] + self.x_offset, decimals=1
                ).item()
                lower_bound_y = torch.round(
                    reduced_traj[n, 1] - self.y_offset, decimals=1
                ).item()
                upper_bound_y = torch.round(
                    reduced_traj[n, 1] + self.y_offset, decimals=1
                ).item()

            # Inverse the grid to pass through RNN
            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Get a perturbation of the activity
            x_0_flow = inverse_grid - states[n, :]

            with torch.no_grad():
                # Return jacobian found from current trajectory
                jac_rec, _ = self.linearization.jacobian(states[n, :])
                # Get next h
                h = states[n, :] + (jac_rec @ x_0_flow.T).T

            # Put next h into a grid format
            cur_region_h = self.reduce_obj.transform(h)
            cur_region_h = torch.tensor(cur_region_h, dtype=self.dtype)

            # Compute velocities between gathered trajectory of grid and original grid values
            x_vel, y_vel = self._compute_velocity(cur_region_h, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            # Reshape data back to grid
            flow_field_list.append(FlowField(x_vel, y_vel, low_dim_grid, speed))

        return flow_field_list

    def _reduce_traj(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Fit PCA object and transform trajectory

        Args:
            trajectory (Tensor): states to reduce
            args (str, ...): regions to gather

        Returns:
            Tensor: reduced states
        """
        # Gather activity for specified region and cell type
        temp_act = torch.reshape(trajectory, (-1, trajectory.shape[-1]))

        # Do PCA on the specified region(s)
        self.reduce_obj.fit(temp_act)
        reduced_traj = self.reduce_obj.transform(temp_act)
        reduced_traj = torch.from_numpy(reduced_traj)

        return reduced_traj

    def _inverse_grid(
        self,
        lower_bound_x: float,
        upper_bound_x: float,
        lower_bound_y: float,
        upper_bound_y: float,
        expand_dims: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Obtain a low dimensional grid and its projection to higher dim state
        space

        Args:
            lower_bound_x (float): lower bound of grid in x direction
            upper_bound_x (float): upper bound of grid in x direction
            lower_bound_y (float): lower bound of grid in y direction
            upper_bound_y (float): upper bound of grid in y direction

        Returns:
            tuple: the low dimensional and projected low dimensional grid
        """
        # Num points is along each axis, not in total
        x = torch.linspace(lower_bound_x, upper_bound_x, self.num_points)
        y = torch.linspace(lower_bound_y, upper_bound_y, self.num_points)

        # Gather 2D grid for flow fields
        xv, yv = torch.meshgrid(x, y, indexing="ij")
        xv, yv = xv.unsqueeze(-1), yv.unsqueeze(-1)

        # Convert the grid to a tensor and flatten for PCA
        low_dim_grid = torch.cat((xv, yv), dim=-1)
        low_dim_grid = torch.flatten(low_dim_grid, start_dim=0, end_dim=1)

        # Inverse PCA to input grid into network
        inverse_grid = self.reduce_obj.inverse_transform(low_dim_grid)
        inverse_grid = inverse_grid.to(self.dtype)

        if expand_dims:
            low_dim_grid = torch.reshape(
                low_dim_grid, (self.num_points, self.num_points, 2)
            )
            inverse_grid = torch.reshape(
                inverse_grid, (self.num_points, self.num_points, inverse_grid.shape[-1])
            )

        return low_dim_grid, inverse_grid

    def _compute_velocity(
        self, h_next: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute velocity, or h_next - h_prev"""
        x_vel = h_next[..., 0] - h_prev[..., 0]
        y_vel = h_next[..., 1] - h_prev[..., 1]
        return x_vel, y_vel

    def _compute_speed(self, x_vel: torch.Tensor, y_vel: torch.Tensor) -> torch.Tensor:
        """compute magnitude of velocities"""
        speed = torch.sqrt(x_vel**2 + y_vel**2)
        return speed / speed.max()

    def _reshape_vals(
        self,
        x_vels: torch.Tensor,
        y_vels: torch.Tensor,
        grid: torch.Tensor,
        speeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Broadcast data to FlowField format

        Args:
            x_vels (Tensor): x velocities
            y_vels (Tensor): y velocities
            grid (Tensor): low dimensional grid coordinates
            speeds (Tensor): magnitude of x_vels and y_vels
        """
        # Reshape to match FlowField object requirements
        x_vels = torch.reshape(x_vels, (self.num_points, self.num_points))
        y_vels = torch.reshape(y_vels, (self.num_points, self.num_points))
        grid = torch.reshape(grid, (self.num_points, self.num_points, 2))
        speeds = torch.reshape(speeds, (self.num_points, self.num_points))
        return x_vels, y_vels, grid, speeds
