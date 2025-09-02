import math

import gpytorch
import torch

from utils import trace_and_save_model

train_x = torch.linspace(0, 1, 100)

train_y = torch.stack(
    [
        torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.5,
    ],
    -1,
)

num_targets = train_y.shape[-1]


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

# --- Add this line after your training loop ---
torch.save(model.state_dict(), "multitask_gp_state.pth")

test_x = torch.linspace(0, 1, 200).unsqueeze(-1)
print("Tracing and saving the model...")
trace_and_save_model(
    model,
    test_x,
    "models",
    "test_independent_multitask_gp_model.pt",
)
