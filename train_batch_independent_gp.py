import gpytorch
import torch
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
):
    """Train a batch independent multitask Gaussian Process model on synthetic data.
    Args:
        data_path (str): Path to the directory containing the NetCDF files.
        target_var (str): Name of the variable to be used as the target.
        sample_size (int): Number of samples to draw from the dataset. The format of this argument depends on the requirements of the chosen sampler.
        training_iterations (int): Number of training iterations.
        learning_rate (float): Learning rate for the optimizer.
        output_dir (str): Directory to save outputs and models.
        min_pfull (float): Minimum pfull value to filter the data.
    """

    ds = NetCDFDataset(
        data_path=data_path,
        # feature_vars=["temp", "qv"],
        feature_vars=["qv"],
        target_var=target_var,
        min_pfull=min_pfull,
        sample_size=sample_size,
    )

    num_pfull = ds.num_pfull
    print(f"Fitting the bottom {num_pfull} atmospheric levels.")

    ### Write sampler code here!
    print("Generating the sampler...")
    sampler = LatinHypercubeSampler(ds, sample_size)

    print("Generating the DataLoader...")
    dl = DataLoader(ds, batch_size=sample_size, sampler=sampler)
    print("Generating batch...")
    train_x, train_y = next(iter(dl))  # ONLY CALL THIS ONCE TO GET A SINGLE BATCH

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
    # test_x_temp = torch.linspace(0, 1, 200)
    # test_x_qv = torch.linspace(0, 1, 200)
    # # test_x = torch.stack([test_x_temp, test_x_qv], -1)
    # test_x = torch.stack([test_x_qv, test_x_qv, test_x_qv], -1)
    test_x = train_x  # Save the GP at the same locations as the training data.
    if torch.cuda.is_available():
        test_x = test_x.cuda()

    # --- Trace the model with TorchScript ---
    print("Tracing and saving the model...")
    trace_and_save_model(
        model,
        test_x,
        output_dir,
        "independent_multitask.pt",
    )


if __name__ == "__main__":
    parser = get_base_parser()

    main(**vars(parser.parse_args()))
