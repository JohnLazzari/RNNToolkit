import torch
import torch.nn as nn
import pytest

from rnntoolkit.linear import Linearization


def test_jacobian_matches_weighted_activation_grad():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="relu")
    rnn.weight_hh_l0 = nn.Parameter(torch.tensor([[0.7, 0.5], [0.4, 0.3]]))
    rnn.weight_ih_l0 = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, -1.0]]))
    rnn.bias_hh_l0 = nn.Parameter(torch.tensor([0.0, 0.0]))
    rnn.bias_ih_l0 = nn.Parameter(torch.tensor([0.0, 0.0]))
    linearization = Linearization(rnn)
    h = torch.tensor([0.2, -0.1])
    x = torch.tensor([1.0, 1.0])

    jac, jac_inp = linearization.jacobian(x, h)

    expected_diag = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    expected_jac = expected_diag @ rnn.weight_hh_l0
    expected_inp = expected_diag @ rnn.weight_ih_l0

    assert torch.allclose(jac, expected_jac)
    assert torch.allclose(jac_inp, expected_inp)


def test_eigenvalues_match_expected():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="relu")
    rnn.weight_hh_l0 = nn.Parameter(torch.tensor([[0.7, 0.5], [0.4, 0.3]]))
    rnn.weight_ih_l0 = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    rnn.bias_hh_l0 = nn.Parameter(torch.tensor([0.0, 0.0]))
    rnn.bias_ih_l0 = nn.Parameter(torch.tensor([0.0, 0.0]))
    linearization = Linearization(rnn)
    h = torch.tensor([0.2, -0.1])
    x = torch.tensor([1.0, 1.0])

    reals, ims, _ = linearization.eigendecomposition(x, h)

    eigenvalues, _ = torch.linalg.eig(rnn.weight_hh_l0)

    # Split real and imaginary parts
    _reals = []
    for eigenvalue in eigenvalues:
        _reals.append(eigenvalue.real.item())
    _reals = torch.tensor(_reals)

    _ims = []
    for eigenvalue in eigenvalues:
        _ims.append(eigenvalue.imag.item())
    _ims = torch.tensor(_ims)

    assert torch.allclose(reals, _reals)
    assert torch.allclose(ims, _ims)
