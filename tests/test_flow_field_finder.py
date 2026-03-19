import torch
import torch.nn as nn
import pytest

from rnntoolkit import FlowFieldFinder


def test_find_nonlinear_flow_returns_flow_fields():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="tanh")

    states = torch.tensor([[0.1, 0.2], [0.2, 0.1], [0.3, -0.1]])
    inp = torch.tensor([[0.0, 0.0], [0.1, 0.2], [-0.1, 0.1]])

    finder = FlowFieldFinder(
        rnn, states, num_points=3, x_offset=1, y_offset=1, x_center=0, y_center=0
    )

    flows = finder.find_nonlinear_flow(states, inp)

    assert len(flows) == states.shape[0]
    assert flows[0].x_vels.shape == (3, 3)
    assert flows[0].grid.shape == (3, 3, 2)


def test_inverse_grid_shapes_match_num_points():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="tanh")
    traj = torch.tensor([[0.0, 0.1], [0.2, 0.3]])
    finder = FlowFieldFinder(
        rnn, traj, num_points=4, x_offset=1, y_offset=1, x_center=0, y_center=0
    )

    low_dim_grid, inv_grid = finder._inverse_grid(-1, 1, -1, 1)

    assert low_dim_grid.shape == (16, 2)
    assert inv_grid.shape == (16, 2)
