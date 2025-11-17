import json
import math
import os
from typing import Any

import gpytorch
import torch
from torch.distributions import (
    AffineTransform,
    MultivariateNormal,
    TransformedDistribution,
)


def trace_model(
    model: gpytorch.models.ExactGP,
    test_x: torch.Tensor,
    output_scale: float = 1.0,
    output_loc: float = 0.0,
):
    """
    Traces a trained GPyTorch ExactGP model, embedding an affine output transform.
    Args:
        model (gpytorch.models.ExactGP): The trained GPyTorch model.
        test_x (torch.Tensor): A sample input tensor for tracing.
        output_scale (float): The scaling factor for the output transformation.
        output_loc (float): The location/shift for the output transformation.
    """
    model.eval()

    class PredictionWrapper(torch.nn.Module):
        def __init__(
            self,
            trained_model: gpytorch.models.ExactGP,
            output_scale: float,
            output_loc: float,
        ):
            super().__init__()
            train_x_input = trained_model.train_inputs[0]
            if train_x_input.dim() != 2:
                raise ValueError(
                    f"train_x must be a 2D tensor. Got {train_x_input.dim()} dims."
                )

            self.register_buffer("train_x", train_x_input)
            self.register_buffer("train_y", trained_model.train_targets)
            self.register_buffer("output_scale", torch.tensor(output_scale))
            self.register_buffer("output_loc", torch.tensor(output_loc))

            base_kernel = trained_model.covar_module.base_kernel
            self.kernel_type = type(base_kernel).__name__

            if self.kernel_type == "RBFKernel":
                self.register_buffer("lengthscale", base_kernel.lengthscale.data)
            elif self.kernel_type == "MaternKernel":
                self.register_buffer("lengthscale", base_kernel.lengthscale.data)
                self.nu = base_kernel.nu
            else:
                raise NotImplementedError(f"Unsupported kernel: {self.kernel_type}")

            self.register_buffer(
                "mean_constant", trained_model.mean_module.constant.data
            )
            self.register_buffer("noise", trained_model.likelihood.noise.data)
            self.register_buffer(
                "outputscale", trained_model.covar_module.outputscale.data
            )

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
                # --- Apply Affine Transformation ---
                # E[a*X + b] = a*E[X] + b
                # Cov[a*X + b] = a^2 * Cov[X]
                final_mean = self.output_scale * pred_mean + self.output_loc
                final_covar = (self.output_scale**2) * pred_covar

                # Return results with consistent batch dimensions
                return final_mean.transpose(0, 1), final_covar.permute(1, 2, 0)

    with torch.no_grad(), gpytorch.settings.trace_mode(True):
        traced_model = torch.jit.trace(
            PredictionWrapper(model, output_scale, output_loc), test_x
        )
    return traced_model


def save_traced_model(traced_model: Any, output_dir: str, name: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    traced_model.save(os.path.join(output_dir, name))
    print(f"Traced model saved to {os.path.join(output_dir, name)}")


def trace_and_save_model(
    model: gpytorch.models.ExactGP,
    test_x: torch.Tensor,
    output_dir: str,
    name: str,
    output_scale: float = 1.0,
    output_loc: float = 0.0,
):
    traced_model = trace_model(model, test_x, output_scale, output_loc)
    save_traced_model(traced_model, output_dir, name)


class TracedGPModelHandler:
    def __init__(self, model_path: str, device: torch.device):
        """
        Loads a traced GP model and its metadata to prepare for inference.
        Args:
            model_path (str): Path to the saved traced model (e.g., 'model.pt').
            device (torch.device): Device to load the model onto.
        """
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        # --- Load Metadata to Infer Inverse Transform ---
        meta_path = model_path.replace(".pt", ".json")
        if not os.path.exists(meta_path):
            print(
                f"Warning: Metadata file not found at {meta_path}. "
                "Assuming identity transform."
            )
            self.metadata = {}
        else:
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

        self.inverse_output_transform = self._get_inverse_transform()

    def _get_inverse_transform(self):
        """Builds the inverse transformation from metadata."""
        config = self.metadata.get("transformations", {}).get("output")
        if not config or config["type"] == "identity":
            # Return the inverse of an identity transform, which is just identity
            return AffineTransform(loc=0.0, scale=1.0).inv
        if config["type"] == "affine":
            # Create the forward transform and return its inverse
            forward_transform = AffineTransform(
                loc=config["loc"], scale=config["scale"]
            )
            return forward_transform.inv
        raise NotImplementedError(f"Unsupported transform: {config['type']}")

    def predict(self, test_x: torch.Tensor, return_original_scale: bool = True):
        """
        Makes predictions and optionally transforms them back to the original scale.
        Args:
            test_x (torch.Tensor): Input tensor for prediction.
            return_original_scale (bool): If True, transforms the prediction back
                                           to the original data scale.
        Returns:
            torch.distributions.TransformedDistribution: The predictive distribution.
        """
        test_x = test_x.to(self.device)

        with torch.no_grad():
            pred_mean_transformed, pred_covar_transformed = self.model(test_x)

        base_distribution = MultivariateNormal(
            loc=pred_mean_transformed.transpose(0, 1),
            covariance_matrix=pred_covar_transformed.permute(2, 0, 1),
        )

        if return_original_scale:
            return TransformedDistribution(
                base_distribution, self.inverse_output_transform
            )
        else:
            return TransformedDistribution(
                base_distribution, AffineTransform(loc=0.0, scale=1.0)
            )
