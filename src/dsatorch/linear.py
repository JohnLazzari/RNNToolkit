import torch
from torch.nn import RNN, GRU, LSTM
import torch.nn.functional as F


class Linearization:
    def __init__(
        self,
        rnn: RNN | GRU | LSTM,
    ):
        """
        Linearization object that stores methods for local analyses of mRNNs

        Args:
            mrnn: mRNN object
        """
        self.rnn = rnn

    def __call__(
        self,
        inp: torch.Tensor,
        h: torch.Tensor,
        delta_inp: torch.Tensor,
        delta_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(inp, h, delta_inp, delta_h)

    @staticmethod
    def relu_grad(x: torch.Tensor) -> torch.Tensor:
        """
        relu function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        # check what this returns
        return torch.autograd.functional.jacobian(F.relu, x)

    @staticmethod
    def tanh_grad(x: torch.Tensor) -> torch.Tensor:
        """
        tanh function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        return torch.autograd.functional.jacobian(F.tanh, x)

    def forward(
        self,
        inp: torch.Tensor,
        h: torch.Tensor,
        delta_inp: torch.Tensor,
        delta_h: torch.Tensor,
        keep_dims=False,
    ) -> torch.Tensor:
        """
        First order taylor exansion of RNN at a given point and input

        Args:
            inp: 1D tensor of input for network at a given state
            h: 1D tensor of network state to linearize about
            delta_inp: perturbation of input
            delta_h: perturbation of state
        """

        # Assert correct shapes
        assert inp.dim() == 1
        assert h.dim() == 1
        assert delta_inp.shape == delta_h.shape

        pert_shape = tuple(delta_inp.shape)

        if delta_inp.dim() > 1:
            delta_inp = delta_inp.flatten(start_dim=0, end_dim=-2)
        if delta_h.dim() > 1:
            delta_h = delta_h.flatten(start_dim=0, end_dim=-2)

        # Get jacobians
        _jacobian, _jacobian_inp = self.jacobian(h)

        # reshape to pass into RNN
        inp = inp.unsqueeze(0).unsqueeze(0)
        h = h.unsqueeze(0).unsqueeze(0)

        # Get h_next for affine function
        _, h_next = self.rnn(inp, h)

        h_pert = (
            h_next.squeeze(0)
            + (_jacobian @ delta_h.T).T
            + (_jacobian_inp @ delta_inp.T).T
        )

        if keep_dims:
            h_pert = torch.reshape(h_pert, pert_shape)

        return h_pert

    def jacobian(
        self, h: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Linearize the dynamics around a state and return the Jacobian.

        Computes the Jacobian of the mRNN update with respect to the hidden state
        evaluated at the provided state ``x`` and (optionally) a subset of regions
        defined by ``*args``. If ``W_inp`` is provided, also returns the Jacobian
        with respect to the input.

        Args:
            x (torch.Tensor): 1D or batched tensor representing the pre-activation state at which to
                linearize (shape ``[H]``).
            *args (str): Optional region names specifying a subset for the Jacobian.
            alpha (float): Discretization factor used in the update.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Jacobian w.r.t. hidden
            state, and optionally (Jacobian w.r.t. input) if ``W_inp`` is provided.
        """
        assert h.dim() == 1

        """
            Taking jacobian of x with respect to F
            In this case, the form should be:
                J_(ij)(x) = -I_(ij) + W_(ij)h'(x_j)
        """

        # Implementing h'(x), diagonalize to multiply by W
        if self.rnn.nonlinearity == "relu":
            d_x_act_diag = self.relu_grad(h)
        elif self.rnn.nonlinearity == "tanh":
            d_x_act_diag = self.tanh_grad(h)
        else:
            raise ValueError("not a valid activation function")

        assert isinstance(self.rnn.weight_hh_l0, torch.Tensor)
        assert isinstance(self.rnn.weight_ih_l0, torch.Tensor)

        # Get final jacobian using form above
        _jacobian = d_x_act_diag @ self.rnn.weight_hh_l0
        _jacobian_inp = d_x_act_diag @ self.rnn.weight_ih_l0
        return _jacobian, _jacobian_inp

    def eigendecomposition(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linearize the network and compute eigen decomposition.

        Args:
            x (torch.Tensor): 1D hidden state where the system is linearized.

        Returns:
            torch.Tensor: Real parts of eigenvalues.
            torch.Tensor: Imag parts of eigenvalues.
            torch.Tensor: Eigenvectors stacked column-wise.
        """
        _jacobian, _ = self.jacobian(x)
        eigenvalues, eigenvectors = torch.linalg.eig(_jacobian)

        # Split real and imaginary parts
        reals = []
        for eigenvalue in eigenvalues:
            reals.append(eigenvalue.real.item())
        reals = torch.tensor(reals)

        ims = []
        for eigenvalue in eigenvalues:
            ims.append(eigenvalue.imag.item())
        ims = torch.tensor(ims)

        return reals, ims, eigenvectors
