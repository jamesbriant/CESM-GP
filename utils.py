import os
from typing import Any

import gpytorch
import torch


def trace_model(
    model: gpytorch.models.ExactGP,
    test_x: torch.Tensor,
):
    class MeanVarModelWrapper(torch.nn.Module):
        def __init__(self, gp):
            super().__init__()
            self.gp = gp

        def forward(self, x):
            output_dist = self.gp(x)
            return output_dist.mean, output_dist.variance

    with (
        torch.no_grad(),
        gpytorch.settings.fast_pred_var(),
        gpytorch.settings.trace_mode(),
    ):
        model.eval()
        traced_model = torch.jit.trace(MeanVarModelWrapper(model), test_x)

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
