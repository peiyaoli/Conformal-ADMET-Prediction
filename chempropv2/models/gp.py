import math
import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

class SparseGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SparseGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.covar_module = InducingPointKernel(self.base_covar_module, 
                                                inducing_points=train_x[:500, :].clone(), 
                                                likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SparseGPRegressor:
    def __init__(self, n_iters: int = 100) -> None:
        self.n_iters = n_iters
        self.optimizer = 
        self.model = SparseGPRegressionModel(train_x, train_y, likelihood)
        


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SparseGPRegressionModel(train_x, train_y, likelihood)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
    
training_iterations = 2 if smoke_test else 100

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train():
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        torch.cuda.empty_cache()

def predict():
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = model.likelihood(model(test_x))