import math
import os
from typing import Any

import gpytorch
import torch


def trace_model(
    model: gpytorch.models.ExactGP,
    test_x: torch.Tensor,
):
    """
    Traces a trained GPyTorch ExactGP model for saving and deployment.

    Args:
        model (gpytorch.models.ExactGP): The trained GPyTorch model.
        test_x (torch.Tensor): A sample input tensor to use for tracing.
                               The shape and dtype must be representative
                               of the inputs the final model will receive.
    """
    # It's crucial to set the model to evaluation mode
    model.eval()

    # This wrapper creates a stateless prediction module.
    # It extracts the trained parameters and rebuilds the prediction logic
    # using simple tensor operations, making it JIT-traceable.
    class PredictionWrapper(torch.nn.Module):
        def __init__(self, trained_model: gpytorch.models.ExactGP):
            super().__init__()
            # --- Enforce 2D Input Convention ---
            train_x_input = trained_model.train_inputs[0]
            if train_x_input.dim() != 2:
                raise ValueError(
                    f"train_x must be a 2D tensor (n_data, n_features). "
                    f"Got {train_x_input.dim()} dimensions."
                )

            # Extract the raw trained parameters (hyperparameters) from the model
            # and store them as buffers.
            self.register_buffer("train_x", train_x_input)
            self.register_buffer("train_y", trained_model.train_targets)

            # --- Kernel Detection and Hyperparameter Extraction ---
            base_kernel = trained_model.covar_module.base_kernel
            self.kernel_type = type(base_kernel).__name__

            if self.kernel_type == "RBFKernel":
                lengthscale = base_kernel.lengthscale.data
                self.register_buffer("lengthscale", lengthscale)
            elif self.kernel_type == "MaternKernel":
                lengthscale = base_kernel.lengthscale.data
                nu = base_kernel.nu
                self.register_buffer("lengthscale", lengthscale)
                self.nu = nu  # Store nu as a simple attribute
            else:
                raise NotImplementedError(
                    f"Kernel type {self.kernel_type} is not supported for tracing."
                )

            # Extract mean, noise, and outputscale (common to all kernels)
            mean_constant = trained_model.mean_module.constant.data
            noise = trained_model.likelihood.noise.data
            outputscale = trained_model.covar_module.outputscale.data

            self.register_buffer("mean_constant", mean_constant)
            self.register_buffer("noise", noise)
            self.register_buffer("outputscale", outputscale)

            # Pre-compute and cache the alpha term, which is central to GP prediction.
            with torch.no_grad():
                K_train_train = self._calculate_kernel_matrix(
                    self.train_x, self.train_x
                )
                K_plus_noise = K_train_train + torch.diag_embed(
                    self.noise.expand(K_train_train.shape[:-1])
                )

                # Reshape y_residual to (num_tasks, num_train_points, 1)
                y_residual = (
                    (self.train_y - self.mean_constant).transpose(0, 1).unsqueeze(-1)
                )

                self.register_buffer(
                    "alpha_cache", torch.linalg.solve(K_plus_noise, y_residual)
                )
                self.register_buffer("K_plus_noise_inv", torch.linalg.inv(K_plus_noise))

        def _calculate_kernel_matrix(self, x1, x2):
            """Dispatcher to the correct stateless kernel function."""
            # Shape of x1: (n, d), x2: (m, d)
            # Resulting kernel shape: (num_tasks, n, m)
            if self.kernel_type == "RBFKernel":
                return self._rbf_kernel(x1, x2)
            elif self.kernel_type == "MaternKernel":
                if self.nu == 0.5:
                    return self._matern12_kernel(x1, x2)
                elif self.nu == 1.5:
                    return self._matern32_kernel(x1, x2)
                elif self.nu == 2.5:
                    return self._matern52_kernel(x1, x2)
                else:
                    raise NotImplementedError(
                        f"Matern kernel with nu={self.nu} is not supported."
                    )
            raise NotImplementedError(
                f"Kernel type {self.kernel_type} is not supported."
            )

        def _rbf_kernel(self, x1, x2):
            """A simple, stateless RBF kernel implementation."""
            dist = torch.cdist(x1, x2, p=2.0).unsqueeze(0)  # Shape: (1, n, m)
            lengthscale = self.lengthscale.view(-1, 1, 1)  # Shape: (num_tasks, 1, 1)
            outputscale = self.outputscale.view(-1, 1, 1)
            return outputscale * torch.exp(-0.5 * (dist.pow(2) / lengthscale.pow(2)))

        def _matern12_kernel(self, x1, x2):
            """Stateless Matérn kernel with nu=1/2 (Exponential)."""
            dist = torch.cdist(x1, x2, p=2.0).unsqueeze(0)
            lengthscale = self.lengthscale.view(-1, 1, 1)
            outputscale = self.outputscale.view(-1, 1, 1)
            return outputscale * torch.exp(-dist / lengthscale)

        def _matern32_kernel(self, x1, x2):
            """Stateless Matérn kernel with nu=3/2."""
            dist = torch.cdist(x1, x2, p=2.0).unsqueeze(0)
            lengthscale = self.lengthscale.view(-1, 1, 1)
            outputscale = self.outputscale.view(-1, 1, 1)
            term1 = math.sqrt(3) * dist / lengthscale
            return outputscale * (1 + term1) * torch.exp(-term1)

        def _matern52_kernel(self, x1, x2):
            """Stateless Matérn kernel with nu=5/2."""
            dist = torch.cdist(x1, x2, p=2.0).unsqueeze(0)
            lengthscale = self.lengthscale.view(-1, 1, 1)
            outputscale = self.outputscale.view(-1, 1, 1)
            term1 = math.sqrt(5) * dist / lengthscale
            term2 = 5 * dist.pow(2) / (3 * lengthscale.pow(2))
            return outputscale * (1 + term1 + term2) * torch.exp(-term1)

        def forward(self, x):
            # Enforce 2D input for predictions
            if x.dim() != 2:
                raise ValueError(
                    f"Input x must be a 2D tensor. Got {x.dim()} dimensions."
                )

            # Check for consistent number of features
            n_features_train = self.train_x.shape[1]
            n_features_test = x.shape[1]
            # The following check may through a warning when traced. This is okay.
            # The check is performing its job perfectly.
            # It acts as a guardrail during the tracing process to ensure that the sample test_x has the correct shape.
            # If it had the wrong shape, the trace would fail with your ValueError, which is exactly what we want.
            if n_features_test != n_features_train:
                raise ValueError(
                    f"Number of features in test data ({n_features_test}) does not "
                    f"match number of features in training data ({n_features_train})."
                )

            n_test = x.shape[0]

            with torch.no_grad():
                # 1. Calculate the prior mean for the test points. Shape: (num_tasks, n_test)
                prior_mean = self.mean_constant.unsqueeze(1).expand(-1, n_test)

                # 2. Calculate the necessary kernel matrices
                K_test_train = self._calculate_kernel_matrix(x, self.train_x)
                # **MODIFIED:** Calculate the full K(x*, x*) matrix
                K_test_test = self._calculate_kernel_matrix(x, x)

                # 3. Calculate the predictive mean. Shape: (num_tasks, n_test)
                pred_mean = prior_mean + K_test_train.matmul(self.alpha_cache).squeeze(
                    -1
                )

                # 4. Calculate the full predictive covariance. Shape: (num_tasks, n_test, n_test)
                solve_term = K_test_train.matmul(self.K_plus_noise_inv)
                pred_covar = K_test_test - solve_term.matmul(
                    K_test_train.transpose(-1, -2)
                )

                # Return results with consistent batch dimensions
                # pred_mean -> (n_test, num_tasks)
                # pred_covar -> (n_test, n_test, num_tasks)
                return pred_mean.transpose(0, 1), pred_covar.permute(1, 2, 0)

    # The gpytorch.settings.trace_mode() is essential. It tells GPyTorch
    # to use operations that are friendly to the JIT tracer.
    with torch.no_grad(), gpytorch.settings.trace_mode(True):
        traced_model = torch.jit.trace(PredictionWrapper(model), test_x)

    return traced_model


def save_traced_model(
    traced_model: Any,
    output_dir: str,
    name: str,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    traced_model.save(os.path.join(output_dir, name))
    print(f"Traced model saved to {os.path.join(output_dir, name)}")


def trace_and_save_model(
    model: gpytorch.models.ExactGP,
    test_x: torch.Tensor,
    output_dir: str,
    name: str,
):
    traced_model = trace_model(model, test_x)
    save_traced_model(traced_model, output_dir, name)
