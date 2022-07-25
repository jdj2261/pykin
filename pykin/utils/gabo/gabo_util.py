import numpy as np
import torch, gpytorch, botorch

from pykin.utils.gabo.module.kernels_torus import TorusProductOfManifoldsRiemannianGaussianKernel
from pykin.utils.gabo.module.robust_trust_regions import TrustRegions


def convert_angle_to_point_torch(thetas):
    """
    Converts the input angle to a point.

    Args:
        thetas (list): actuated joint names
    
    Returns:
        thetas (dict): Dictionary of actuated joint angles
    """
    point = []
    for theta in thetas:
        point.append(np.cos(theta))
        point.append(np.sin(theta))
    return torch.tensor(point)

def convert_point_to_angle_torch(point):
    """
    Converts the input torus point tensor to an angle tensor using atan2.

    Args:
        point (tensor): Input torus point tensor
    
    Returns:
        angle (tensor): Converted angle tensor
    """
    angle_set = []
    point_set = point.view(-1, 2)
    for point in point_set:
        angle = torch.atan2(point[1], point[0])
        angle_set.append(angle)
    return torch.tensor(angle_set)

def get_bounds(dimension):
    """
    Get optimize bound for bayesian optimization
    """
    bounds = torch.stack([-torch.ones(2 * dimension, dtype=torch.float64),
                        torch.ones(2 * dimension, dtype=torch.float64)])
    return bounds

def init_gp_model(opt_dimension, device, x_data, y_data):
    """
    Initialize Gaussian Process model
    """
    # Define base kernel
    base_kernel = TorusProductOfManifoldsRiemannianGaussianKernel(dim=opt_dimension)
    # Define bounds
    bounds = get_bounds(dimension=opt_dimension)
    # Define kernel function
    k_fct = gpytorch.kernels.ScaleKernel(base_kernel,
                                        outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    k_fct.to(device)

    # A constant mean function is already included in the model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)    
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_fct = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                        noise_constraint=
                                                                        gpytorch.constraints.GreaterThan(1e-8),
                                                                        initial_value=noise_prior_mode)
    lik_fct.to(device)
    model = botorch.models.SingleTaskGP(x_data, y_data[:, None], covar_module=k_fct, likelihood=lik_fct)
    model.to(device)

    # Define the marginal log-likelihood
    mll_fct = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    mll_fct.to(device)

    # Define the constraints and processing functions in function of the BO type and of the manifold
    constraints = None

    # Define solver
    solver = TrustRegions(maxiter=200)

    return mll_fct, model, solver, bounds, constraints