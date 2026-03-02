import torch
import torch.nn as nn
import pytest

from rnntoolkit.linear import Linearization


def test_relu_grad_is_diagonal():
    x = torch.tensor([1.0, -1.0])
    jac = Linearization.relu_grad(x)
    expected = torch.diag(torch.tensor([1.0, 0.0]))
    assert torch.equal(jac, expected)


def test_tanh_grad_matches_derivative():
    x = torch.tensor([0.25, -0.5])
    jac = Linearization.tanh_grad(x)
    expected = torch.diag(1 - torch.tanh(x) ** 2)
    assert torch.allclose(jac, expected)


def test_jacobian_matches_weighted_activation_grad():
    rnn = nn.RNN(input_size=2, hidden_size=2, batch_first=True, nonlinearity="tanh")
    linearization = Linearization(rnn)
    h = torch.tensor([0.2, -0.1])

    jac, jac_inp = linearization.jacobian(h)

    expected_diag = torch.diag(1 - torch.tanh(h) ** 2)
    expected_jac = expected_diag @ rnn.weight_hh_l0
    expected_inp = expected_diag @ rnn.weight_ih_l0

    assert torch.allclose(jac, expected_jac)
    assert torch.allclose(jac_inp, expected_inp)
