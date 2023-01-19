import gpytorch
import torch
from utils import gen_X_set
from sklearn.metrics import f1_score

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def model_training(X, y, lr, iter):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = likelihood.cuda() if torch.cuda.is_available() else likelihood
    model = ExactGPModel(X, y, likelihood)
    model = model.cuda() if torch.cuda.is_available() else model
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    model = model.float()
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=lr)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X.float())
        # Calc loss and backprop gradients
        loss = -mll(output, y.float())
        loss.mean().backward()
        optimizer.step()
    return model, likelihood

def sample_from_GP(X_range, N, model):
    X_sample = gen_X_set(X_range, N)
    f_preds = model(X_sample.float())
    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    y_sample = f_preds.sample(sample_shape=torch.Size((1,))).reshape(N)
    return X_sample, y_sample

def evaluate_lse_model(model, likelihood, X, y, h):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    with torch.no_grad():
        # Make predictions
        observed_pred = likelihood(model(X.float()))
        lower, _ = observed_pred.confidence_region()
        lse_pred = lower >= h
        lse_y = y >= h
        f1 = f1_score(lse_y.cpu(), lse_pred.cpu())
        return f1