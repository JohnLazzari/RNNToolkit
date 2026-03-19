import torch

from rnntoolkit import FlowField


def _make_flow_field():
    x_vels = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    y_vels = x_vels + 1
    speeds = torch.ones(2, 3)
    grid = torch.zeros(2, 3, 2)
    return FlowField(x_vels=x_vels, y_vels=y_vels, grid=grid, speeds=speeds)


def test_len_counts_grid_cells():
    flow = _make_flow_field()
    assert len(flow) == 6


def test_getitem_int_preserves_2d_shape():
    flow = _make_flow_field()
    sub = flow[1]
    assert sub.x_vels.shape == (1, 3)
    assert sub.grid.shape == (1, 3, 2)


def test_getitem_slice_tuple_broadcasts():
    flow = _make_flow_field()
    sub = flow[:, 1]
    assert sub.x_vels.shape == (2, 1)
    assert sub.grid.shape == (2, 1, 2)
