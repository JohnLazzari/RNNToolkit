import torch
import torch.nn as nn

from rnntoolkit import FixedPointFinder
from rnntoolkit import FixedPointCollection


def test_sample_states_excludes_zero_rows():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="tanh")
    finder = FixedPointFinder(rnn, verbose=False, super_verbose=False)

    state_traj = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    sampled = finder.sample_states(state_traj, n_inits=1, exclude_zero_tensors=True)

    assert torch.allclose(sampled[0], torch.tensor([1.0, -1.0]))


def test_identify_q_outliers_and_non_outliers():
    xstar = torch.zeros(3, 2)
    fps = FixedPointCollection(
        xstar=xstar,
        qstar=torch.tensor([0.1, 1.0, 0.2]),
        n_iters=torch.tensor([1, 1, 1]),
    )

    out_idx = FixedPointFinder.identify_q_outliers(fps, q_thresh=0.5)
    non_out_idx = FixedPointFinder.identify_q_non_outliers(fps, q_thresh=0.5)

    assert torch.equal(out_idx, torch.tensor([1]))
    assert torch.equal(non_out_idx, torch.tensor([0, 2]))


def test_broadcast_nxd_handles_1d_and_2d():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="tanh")
    finder = FixedPointFinder(rnn, verbose=False, super_verbose=False)

    one_d = torch.tensor([1.0, 2.0])
    tiled = finder._broadcast_nxd(one_d, tile_n=3)
    assert tiled.shape == (3, 2)

    two_d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    flattened = finder._broadcast_nxd(two_d)
    assert flattened.shape == (2, 2)


def test_find_fixed_points_smoke():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="tanh")
    finder = FixedPointFinder(
        rnn,
        max_iters=1,
        tol_q=1e-6,
        tol_dq=1e-6,
        verbose=False,
        super_verbose=False,
    )

    initial_states = torch.tensor([[0.1, -0.1], [0.2, 0.0]])
    ext_inputs = torch.tensor([0.0, 0.0])

    unique_fps, all_fps = finder.find_fixed_points(initial_states, ext_inputs)

    assert unique_fps.n > 0
    assert all_fps.n == initial_states.shape[0]
