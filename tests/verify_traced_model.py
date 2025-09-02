import math

import gpytorch
import torch
from matplotlib import pyplot as plt

# This script verifies that a JIT-traced model produces the same output
# as the original GPyTorch model from which it was derived.

# --- Step 1: Recreate the Model and Data ---
# This section must be identical to the training script to ensure the model
# architecture and data are consistent.

# Regenerate the exact same training data, ensuring both tasks are active.
train_x = torch.linspace(0, 1, 100).view(-1, 1)  # Shape (100, 1)
train_y = torch.stack(
    [
        torch.sin(train_x * (2 * math.pi)).squeeze(-1)
        + torch.randn(train_x.size(0)) * 0.2,
        torch.cos(train_x * (2 * math.pi)).squeeze(-1)
        + torch.randn(train_x.size(0)) * 0.2,
    ],
    -1,
)  # Shape (100, 2)

num_tasks = train_y.shape[-1]


# Redefine the model class exactly as in the training script
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_tasks])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


# --- Step 2: Load Both the Original and Traced Models ---

# Load the original model's trained state (hyperparameters)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
original_model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)

state_dict = torch.load("multitask_gp_state.pth")
original_model.load_state_dict(state_dict)

# Load the JIT-traced model
# Make sure the filename matches the one used in your training script
traced_model = torch.jit.load("models/test_independent_multitask_gp_model.pt")


# --- Step 3: Generate Predictions from Both Models ---

# Set models to evaluation mode
original_model.eval()
likelihood.eval()
traced_model.eval()

# Create test points
test_x = torch.linspace(0, 1, 51).view(-1, 1)  # Shape (51, 1)

# Predictions from the original GPyTorch model
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # original_pred_dist = likelihood(original_model(test_x))
    original_pred_dist = original_model(test_x)
    original_mean = original_pred_dist.mean
    original_lower, original_upper = original_pred_dist.confidence_region()

# Predictions from the traced model
with torch.no_grad():
    traced_mean, traced_covar = traced_model(test_x)
    # Get variance from the diagonal of the covariance matrix
    traced_var = traced_covar.diagonal(dim1=0, dim2=1).T
    traced_stddev = torch.sqrt(traced_var)
    traced_lower = traced_mean - 2 * traced_stddev
    traced_upper = traced_mean + 2 * traced_stddev


# --- Step 4: Plot the Comparison ---

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
task_titles = ["Task 1: sin(x)", "Task 2: cos(x)"]

for i, ax in enumerate(axes):
    # Plot training data
    ax.plot(train_x.numpy(), train_y[:, i].numpy(), "k*", label="Observed Data")

    # Plot original model's prediction
    ax.plot(
        test_x.numpy(), original_mean[:, i].numpy(), "b", label="Original Model Mean"
    )
    ax.fill_between(
        test_x.numpy().flatten(),
        original_lower[:, i].numpy(),
        original_upper[:, i].numpy(),
        alpha=0.3,
        color="blue",
        label="Original Model Confidence",
    )

    # Plot traced model's prediction (should overlay perfectly)
    ax.plot(test_x.numpy(), traced_mean[:, i].numpy(), "r--", label="Traced Model Mean")
    ax.fill_between(
        test_x.numpy().flatten(),
        traced_lower[:, i].numpy(),
        traced_upper[:, i].numpy(),
        alpha=0.3,
        color="red",
        label="Traced Model Confidence",
    )

    ax.set_title(task_titles[i])
    ax.legend()
    ax.set_ylim([-3, 3])

plt.suptitle("Comparison of Original and Traced Model Predictions")
plt.tight_layout()
plt.show()
