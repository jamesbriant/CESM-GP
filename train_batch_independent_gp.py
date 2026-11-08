import json
import os
from datetime import datetime, timezone

import gpytorch
import torch
from torch.distributions import AffineTransform
from torch.utils.data import DataLoader

from argparser import get_base_parser
from dataset import NetCDFDataset
from samplers import LatinHypercubeSampler
from tracer import trace_and_save_model


def main(
    data_path: str,
    target_var: str,
    sample_size: int,
    training_iterations: int,
    learning_rate: float,
    output_dir: str,
    min_pfull: float = 0,
    target_scale: float = 1.0,
    target_loc: float = 0.0,
    seed: int = 2025,
):
    """Train a batch independent multitask Gaussian Process model on synthetic data."""
    # --- Reproducibility ---
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Random seed set to {seed}")

    # --- Load Dataset ---
    ds = NetCDFDataset(
        data_path=data_path,
        feature_vars=["temp", "qv"],
        target_var=target_var,
        min_pfull=min_pfull,
        sample_size=sample_size,
    )

    num_pfull = ds.num_pfull
    print(f"Fitting the bottom {num_pfull} atmospheric levels.")

    print("Generating the sampler...")
    sampler = LatinHypercubeSampler(ds, sample_size)

    print("Generating the DataLoader...")
    dl = DataLoader(ds, batch_size=sample_size, sampler=sampler)
    print("Generating batch...")
    train_x, train_y = next(iter(dl))

    # --- Apply Transformations ---
    output_transform = AffineTransform(loc=target_loc, scale=target_scale)
    train_y = output_transform(train_y)

    print(f"train_x shape: {train_x.shape}")
    print(f"train_y shape: {train_y.shape}")

    class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(
                batch_shape=torch.Size([num_pfull])
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_pfull])),
                batch_shape=torch.Size([num_pfull]),
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_pfull)
    model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)

    # --- Use the GPU if available ---
    if torch.cuda.is_available():
        print("Using CUDA")
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    print("Buliding the optimizer...")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print("Training the model...")
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    print("Finished training!")

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # --- Design the test points ---
    # Save the GP at the same locations as the training data.
    # There is perhaps an opportunity to save memory here.
    if torch.cuda.is_available():
        train_x = train_x.cuda()

    # --- Trace and Save ---
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    model_name = f"{timestamp}_{target_var}_pfull{min_pfull}_samples{sample_size}"
    meta_path = os.path.join(output_dir, f"{model_name}.json")

    print("Tracing and saving the model...")
    trace_and_save_model(
        model,
        train_x,
        output_dir,
        f"{model_name}.pt",
        output_scale=target_scale,
        output_loc=target_loc,
    )

    # --- Save Metadata for record-keeping ---
    metadata = {
        "model_name": model_name,
        "target_variable": target_var,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_args": {
            "sample_size": sample_size,
            "training_iterations": training_iterations,
            "learning_rate": learning_rate,
            "min_pfull": min_pfull,
            "seed": seed,
        },
        "transformations": {
            "output": {
                "type": "affine",
                "scale": target_scale,
                "loc": target_loc,
            }
        },
        "training_indices": ds.sampled_idxes,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    parser = get_base_parser()

    main(**vars(parser.parse_args()))
