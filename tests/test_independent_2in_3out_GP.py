import math

import gpytorch
import torch
from matplotlib import pyplot as plt

from tracer import trace_and_save_model

train_x0 = torch.linspace(0, 1, 100)
train_x1 = torch.linspace(0.4, 0.8, 100)
train_x = torch.stack([train_x0, train_x1], -1)


train_y = torch.stack(
    [
        torch.sin(train_x0 * (2 * math.pi) + train_x1)
        + torch.randn(train_x0.size()) * 0.2,
        torch.cos(train_x0 * (2 * math.pi) + train_x1)
        + torch.randn(train_x0.size()) * 0.5,
        torch.sin(train_x0 * (4 * math.pi) + train_x1)
        + torch.randn(train_x0.size()) * 0.3,
    ],
    -1,
)

num_targets = train_y.shape[-1]

print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_targets])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_targets])),
            batch_shape=torch.Size([num_targets]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_targets)
model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 50
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

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # test_x = torch.linspace(0, 1, 51)
    test_x1 = torch.linspace(0, 1, 51)
    test_x2 = torch.linspace(0.4, 0.8, 51)
    test_x = torch.stack([test_x1, test_x2], -1)
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task

print(mean.shape)
print(lower.shape)
print(upper.shape)

# Initialize plots
f, axes = plt.subplots(num_targets, 2, figsize=(8, 3 * num_targets))
for i in range(num_targets):
    ax = axes[i] if num_targets > 1 else axes
    ax[0].plot(test_x[:, 0].numpy(), mean[:, i].numpy())
    # ax[0].fill_between(
    #     test_x.numpy(),
    #     lower[:, i].numpy(),
    #     upper[:, i].numpy(),
    #     alpha=0.5,
    #     color="lightblue",
    # )
    ax[0].set_title(f"Task {i + 1} Predictions")
    # ax[0].set_ylim([-3, 3])
    ax[1].plot(test_x[:, 1].numpy(), mean[:, i].numpy())
    ax[1].set_title(f"Task {i + 1} Predictions")
    # ax[1].set_ylim([-3, 3])

plt.tight_layout()
plt.show()

print("Tracing and saving the model...")
trace_and_save_model(
    model,
    test_x,
    "models",
    "test_independent_2in_3out.pt",
)
