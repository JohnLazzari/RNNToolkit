import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from rnntoolkit.linear import Linearization
from rnntoolkit.flow_fields.flow_field import FlowField
from typing import Generic, TypeVar, Tuple

RNN = TypeVar("RNN", bound=nn.Module)


class FlowFieldFinderBase(Generic[RNN]):
    def __init__(
        self,
        rnn: RNN,
        num_points: int,
        x_offset: int,
        y_offset: int,
        x_center: int,
        y_center: int,
        **kwargs,
    ):
        """
        Flow field that gathers a flow field about a specified trajectory

        Args:
            mrnn (RNN): RNN-like object
            num_points (int): number of points to use in grid, results in (num_points, num_points)
            x_offset (int): scale to offset grid about trajectory in x direction
            y_offset (int): scale to offset grid about trajectory in y direction
            follow_traj (bool): whether or not to center the grid around each trajectory
        """
        self.rnn = rnn
        self.num_points = num_points
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_center = x_center
        self.y_center = y_center
        self.time_dim = 1 if self.rnn.batch_first else 0
        self.dtype = next(self.rnn.parameters()).dtype

        # class objects
        self.reduce_obj = PCA(n_components=2)

    def find_nonlinear_flow(self, *args, **kwargs) -> list:
        """Compute 2D flow fields in a region subspace along a trajectory."""
        raise NotImplementedError

    def find_linear_flow(self, *args, **kwargs) -> list:
        """Compute linearized flow fields in a 2D subspace."""
        raise NotImplementedError

    def _nxd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Broadcast to nxd, even for a 1d tensor

        Args:
            x (Tensor): tensor to broadcast
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.flatten(x, end_dim=-2)
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

    def _set_bounds(self) -> Tuple[float, float, float, float]:
        lower_bound_x = self.x_center - self.x_offset
        upper_bound_x = self.x_center + self.x_offset
        lower_bound_y = self.y_center - self.y_offset
        upper_bound_y = self.y_center + self.y_offset
        return lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y

    def _set_tv_bounds(self, traj: torch.Tensor) -> Tuple[float, float, float, float]:
        lower_bound_x = torch.round(traj[0] - self.x_offset, decimals=1).item()
        upper_bound_x = torch.round(traj[0] + self.x_offset, decimals=1).item()
        lower_bound_y = torch.round(traj[1] - self.y_offset, decimals=1).item()
        upper_bound_y = torch.round(traj[1] + self.y_offset, decimals=1).item()
        return lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y
