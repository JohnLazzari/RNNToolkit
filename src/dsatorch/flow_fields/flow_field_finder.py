import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from dsatorch.linear import Linearization
from dsatorch.flow_fields.flow_field import FlowField
from typing import Generic, TypeVar, Tuple

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
            mrnn (RNN): RNN-like object
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
        self.time_dim = 1 if self.rnn.batch_first else 0
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

        # Unload kwargs
        traj_to_reduce = (
            kwargs["traj_to_reduce"] if "traj_to_reduce" in kwargs else states
        )

        flow_field_list = []

        # Reshape to nxd
        states, inp = self._nxd(states), self._nxd(inp)

        # flatten in case the dimensions are larger than 2
        states = torch.flatten(states, end_dim=-2)
        inp = torch.flatten(inp, end_dim=-2)

        # assert states and input match shape
        assert states.shape[0] == inp.shape[0]
        n_states = states.shape[0]

        self._fit_traj(traj_to_reduce)
        reduced_traj = self._reduce_traj(states)

        # Now going through trajectory
        for n in range(n_states):
            flow_field = self._compute_nonlinear_flowfield(reduced_traj[n], inp[n])
            flow_field_list.append(flow_field)

        return flow_field_list

    def find_linear_flow(
        self, states: torch.Tensor, inp: torch.Tensor, inp_next: torch.Tensor, **kwargs
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

        # Unload kwargs
        traj_to_reduce = (
            kwargs["traj_to_reduce"] if "traj_to_reduce" in kwargs else states
        )

        # reshape to nxd
        states, inp, inp_next = self._nxd(states), self._nxd(inp), self._nxd(inp_next)

        # flatten in case the dimensions are larger than 2
        states = torch.flatten(states, end_dim=-2)
        inp = torch.flatten(inp, end_dim=-2)
        inp_next = torch.flatten(inp_next, end_dim=-2)

        assert inp.shape[0] == inp_next.shape[0]
        assert states.shape[0] == inp.shape[0]
        n_states = states.shape[0]

        # Lists for x and y velocities
        flow_field_list = []

        # Reduce the regional trajectories and return pca object
        self._fit_traj(traj_to_reduce)
        reduced_traj = self._reduce_traj(states)

        for n in range(n_states):
            flow_field = self._compute_linear_flowfield(
                states[n], reduced_traj[n], inp[n], inp_next[n]
            )
            flow_field_list.append(flow_field)

        return flow_field_list

    def _compute_nonlinear_flowfield(self, reduced_traj_n, inp_n):
        # If follow trajectory is true get grid centered around current t
        # This will make a different grid for each state (n grids)
        if self.follow_traj:
            lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                self._set_tv_bounds(reduced_traj_n)
            )
        else:
            lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                self._set_bounds(center=0)
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

        return FlowField(x_vel, y_vel, low_dim_grid, speed)

    def _compute_linear_flowfield(self, states_n, reduced_traj_n, inp_n, inp_next_n):
        # If follow trajectory is true get grid centered around current t
        # This will make a different grid for each state (n grids)
        if self.follow_traj:
            lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                self._set_tv_bounds(reduced_traj_n)
            )
        else:
            lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                self._set_bounds(center=0)
            )

        # Inverse the grid to pass through RNN
        low_dim_grid, inverse_grid = self._inverse_grid(
            lower_bound_x,
            upper_bound_x,
            lower_bound_y,
            upper_bound_y,
        )

        # Get a perturbation of the activity
        x_0_flow = inverse_grid - states_n
        delta_inp = inp_next_n - inp_n

        with torch.no_grad():
            # Return jacobian found from current trajectory
            jac_rec, jac_inp = self.linearization.jacobian(states_n)
            # Get next h
            h = states_n + (jac_rec @ x_0_flow.T).T + (jac_inp @ delta_inp.T).T

        # Put next h into a grid format
        h_next = self._reduce_traj(h)

        # Compute velocities between gathered trajectory of grid and original grid values
        x_vel, y_vel = self._compute_velocity(h_next, low_dim_grid)
        speed = self._compute_speed(x_vel, y_vel)

        x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
            x_vel, y_vel, low_dim_grid, speed
        )

        # Reshape data back to grid
        return FlowField(x_vel, y_vel, low_dim_grid, speed)

    def _nxd(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def _fit_traj(self, trajectory: torch.Tensor):
        """
        Fit PCA object

        Args:
            trajectory (Tensor): states to reduce
        """
        # Gather activity for specified region and cell type
        temp_act = torch.reshape(trajectory, (-1, trajectory.shape[-1]))
        # Do PCA on the specified region(s)
        self.reduce_obj.fit(temp_act)

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

    def _set_bounds(self, center: float = 0.0) -> Tuple[float, float, float, float]:
        lower_bound_x = center - self.x_offset
        upper_bound_x = center + self.x_offset
        lower_bound_y = center - self.y_offset
        upper_bound_y = center + self.y_offset
        return lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y

    def _set_tv_bounds(self, traj: torch.Tensor) -> Tuple[float, float, float, float]:
        lower_bound_x = torch.round(traj[0] - self.x_offset, decimals=1).item()
        upper_bound_x = torch.round(traj[0] + self.x_offset, decimals=1).item()
        lower_bound_y = torch.round(traj[1] - self.y_offset, decimals=1).item()
        upper_bound_y = torch.round(traj[1] + self.y_offset, decimals=1).item()
        return lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y
