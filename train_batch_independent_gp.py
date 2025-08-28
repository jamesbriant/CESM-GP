from typing import Any

import gpytorch
import torch
from torch.utils.data import DataLoader

from argparser import get_base_parser
from dataset import NetCDFDataset
from samplers import LatinHypercubeSampler
from utils import trace_and_save_model


def main(
    data_path: str,
    target_var: str,
    sample_size: Any,
    training_iterations: int,
    learning_rate: float,
    output_dir: str,
    min_pfull: float = 0,
):
    """Train a batch independent multitask Gaussian Process model on synthetic data.
    Args:
        data_path (str): Path to the directory containing the NetCDF files.
        target_var (str): Name of the variable to be used as the target.
        sample_size (Any): Number of samples to draw from the dataset. The format of this argument depends on the requirements of the chosen sampler.
        training_iterations (int): Number of training iterations.
        learning_rate (float): Learning rate for the optimizer.
        output_dir (str): Directory to save outputs and models.
        min_pfull (float): Minimum pfull value to filter the data.
    """

    ds = NetCDFDataset(
        data_path=data_path,
        feature_vars=["temp", "qv"],
        target_var=target_var,
        min_pfull=min_pfull,
    )

    num_pfull = ds.num_pfull
    print(f"Fitting the bottom {num_pfull} atmospheric levels.")

    ### Write sampler code here!
    n_samples = 100  # example value
    sampler = LatinHypercubeSampler(ds, n_samples)

    dl = DataLoader(ds, batch_size=n_samples, shuffle=False, sampler=sampler)
    train_x, train_y = next(iter(dl))  # ONLY CALL THIS ONCE TO GET A SINGLE BATCH

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
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    # # Initialize plots
    # f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

    test_x = torch.linspace(0, 1, 51)
    if torch.cuda.is_available():
        test_x = test_x.cuda()

    # # Make predictions
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     predictions = likelihood(model(test_x))
    #     mean = predictions.mean
    #     lower, upper = predictions.confidence_region()

    # if torch.cuda.is_available():
    #     mean = mean.cpu()
    #     lower = lower.cpu()
    #     upper = upper.cpu()

    #     train_x = train_x.cpu()
    #     train_y = train_y.cpu()
    #     test_x = test_x.cpu()

    # # This contains predictions for both tasks, flattened out
    # # The first half of the predictions is for the first task
    # # The second half is for the second task

    # # Plot training data as black stars
    # y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), "k*")
    # # Predictive mean as blue line
    # y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), "b")
    # # Shade in confidence
    # y1_ax.fill_between(
    #     test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5
    # )
    # y1_ax.set_ylim([-3, 3])
    # y1_ax.legend(["Observed Data", "Mean", "Confidence"])
    # y1_ax.set_title("Observed Values (Likelihood)")

    # # Plot training data as black stars
    # y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), "k*")
    # # Predictive mean as blue line
    # y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), "b")
    # # Shade in confidence
    # y2_ax.fill_between(
    #     test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5
    # )
    # y2_ax.set_ylim([-3, 3])
    # y2_ax.legend(["Observed Data", "Mean", "Confidence"])
    # y2_ax.set_title("Observed Values (Likelihood)")

    # # plt.show()
    # if not os.path.exists("figures"):
    #     os.makedirs("figures")
    # plt.savefig("figures/independent_multitask_gp.png")

    # --- Trace the model with TorchScript ---
    trace_and_save_model(
        model,
        test_x,
        output_dir,
        "independent_multitask_gp_model.pt",
    )


if __name__ == "__main__":
    parser = get_base_parser()

    main(**vars(parser.parse_args()))
