import torch
import torch.nn as nn


class Linearization:
    def __init__(
        self,
        rnn: nn.RNN,
    ):
        """
        Linearization object that stores methods for local analyses of mRNNs

        Args:
            mrnn: mRNN object
        """
        self.rnn = rnn

    def __call__(
        self,
        input: torch.Tensor,
        h: torch.Tensor,
        delta_input: torch.Tensor,
        delta_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(input, h, delta_input, delta_h)

    def forward(
        self,
        input: torch.Tensor,
        h: torch.Tensor,
        delta_input: torch.Tensor,
        delta_h: torch.Tensor,
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
        assert input.dim() == 1
        assert h.dim() == 1

        if delta_h.dim() > 1:
            delta_h = delta_h.flatten(start_dim=0, end_dim=-2)

        # Get jacobians
        _jacobian, _jacobian_inp = self.jacobian(input, h)

        # reshape to pass into RNN
        input = input.unsqueeze(0).unsqueeze(0)
        h = h.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # Get h_next for affine function
            _, h_next = self.rnn(input, h)

        h_pert = (
            h_next.squeeze(0)
            + (_jacobian @ delta_h.T).T
            + (_jacobian_inp @ delta_input)
        )

        return h_pert

    def jacobian(
        self, input: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        assert input.dim() == 1

        input = input.unsqueeze(0).unsqueeze(0)
        h = h.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            _, _jacobians_h = torch.autograd.functional.jacobian(self.rnn, (input, h))

        _jacobian_input, _jacobian_h = _jacobians_h

        return _jacobian_h.squeeze(), _jacobian_input.squeeze()

    def eigendecomposition(
        self, input: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linearize the network and compute eigen decomposition.

        Args:
            x (torch.Tensor): 1D hidden state where the system is linearized.

        Returns:
            torch.Tensor: Real parts of eigenvalues.
            torch.Tensor: Imag parts of eigenvalues.
            torch.Tensor: Eigenvectors stacked column-wise.
        """
        _jacobian, _ = self.jacobian(input, h)
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
