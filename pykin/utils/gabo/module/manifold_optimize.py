"""
Original code is part of the MaternGaBO library.
Github repo : https://github.com/NoemieJaquier/MaternGaBO
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
"""

import numpy as np
import types
import torch
from torch import Tensor
from torch.nn import Module

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.utils import is_nonnegative
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.generation import get_best_candidates
from botorch.optim.initializers import initialize_q_batch, initialize_q_batch_nonneg

from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds.product import Product
from pymanopt.solvers.solver import Solver

import pymanopt
from pymanopt import Problem

from pykin.utils.gabo.module.approximate_hessian import get_hessianfd

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'
    
torch.set_default_dtype(torch.float32)


# This function is based on the botorch.optim.joint_optimize function of botorch.
def joint_optimize_manifold(
    acq_function: AcquisitionFunction,
    manifold: Manifold,
    solver: Solver,
    q: int,
    num_restarts: int,
    raw_samples: int,
    bounds: Tensor,
    sample_type: torch.dtype = torch.float64,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
    inequality_constraints: Optional[List[Callable]] = None,
    equality_constraints: Optional[List[Callable]] = None,
    pre_processing_manifold: Optional[Callable[[Tensor], Tensor]] = None,
    post_processing_manifold: Optional[Callable[[Tensor], Tensor]] = None,
    approx_hessian: bool = False,
    solver_init_conds: bool = False,
) -> Tensor:
    """
    This function generates a set of candidates via joint multi-start optimization

    Parameters
    ----------
    :param acq_function: the acquisition function
    :param manifold: the manifold in the optimization takes place (pymanopt manifold)
    :param solver: solver on manifold to solve the optimization (pymanopt solver)
    :param q: number of candidates
    :param num_restarts: number of starting points for multistart acquisition function optimization
    :param raw_samples: number of samples for initialization
    :param bounds: a `2 x d` tensor of lower and upper bounds for each column of `X`
    :param sample_type: type of the generated samples for initialization

    Optional parameters
    -------------------
    :param options: options for candidate generation
    :param inequality_constraints: inequality constraint or list of inequality constraints, satisfied if >= 0
    :param equality_constraints: equality constraint or list of equality constraints, satisfied if = 0
    :param pre_processing_manifold: a function that pre-process the data on the manifold before the optimization.
        Typically, this can be used to transform vectors (required by the GP) to the corresponding matrices (required
        for matrix-manifold optimization)
    :param post_processing_manifold: a function that post-process the data on the manifold after the optimization.
        Typically, this can be used to transform matrices (required for matrix-manifold optimization) to vectors
        (required by the GP).
    :param approx_hessian: if True, the Hessian of the cost is approximated with finite differences of the gradient
    :param solver_init_conds: if True, the initialization is made inside the solver. This has to be True for
        population-based methods, e.g. PSO, Nelder mead.

    Returns
    -------
    :return: a `q x d` tensor of generated candidates.
    """

    options = options or {}
    batch_initial_conditions = \
        gen_batch_initial_conditions_manifold(acq_function=acq_function, manifold=manifold, bounds=bounds,
                                              q=None if isinstance(acq_function, AnalyticAcquisitionFunction) else q,
                                              num_restarts=num_restarts, raw_samples=raw_samples,
                                              sample_type=sample_type,
                                              options=options, post_processing_manifold=post_processing_manifold)

    batch_limit = options.get("batch_limit", num_restarts)
    batch_candidates_list = []
    batch_acq_values_list = []
    start_idx = 0
    while start_idx < num_restarts:
        end_idx = min(start_idx + batch_limit, num_restarts)
        # optimize using random restart optimization
        batch_candidates_curr, batch_acq_values_curr = \
            gen_candidates_manifold(initial_conditions=batch_initial_conditions[start_idx:end_idx],
                                    acquisition_function=acq_function, manifold=manifold, solver=solver,
                                    pre_processing_manifold=pre_processing_manifold,
                                    post_processing_manifold=post_processing_manifold,
                                    lower_bounds=bounds[0], upper_bounds=bounds[1],
                                    options={k: v for k, v in options.items()
                                             if k not in ("batch_limit", "nonnegative")},
                                    inequality_constraints=inequality_constraints,
                                    equality_constraints=equality_constraints,
                                    approx_hessian=approx_hessian, solver_init_conds=solver_init_conds)

        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        start_idx += batch_limit

    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)
    return get_best_candidates(batch_candidates=batch_candidates, batch_values=batch_acq_values)


# This function is based on the botorch.gen.gen_candidates_scipy
def gen_candidates_manifold(
        initial_conditions: Tensor,
        acquisition_function: Module,
        manifold: Manifold,
        solver: Solver,
        pre_processing_manifold: Optional[Callable[[Tensor], Tensor]] = None,
        post_processing_manifold: Optional[Callable[[Tensor], Tensor]] = None,
        lower_bounds: Optional[Union[float, Tensor]] = None,
        upper_bounds: Optional[Union[float, Tensor]] = None,
        inequality_constraints: Optional[List[Callable]] = None,
        equality_constraints: Optional[List[Callable]] = None,
        approx_hessian: bool = False,
        solver_init_conds: bool = False,
        options: Optional[Dict[str, Union[bool, float, int]]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    This function generates a set of candidates using `scipy.optimize.minimize`

    Parameters
    ----------
    :param initial_conditions: starting points for optimization
    :param acquisition_function: acquisition function to be optimized
    :param manifold: the manifold in the optimization takes place (pymanopt manifold)
    :param solver: solver on manifold to solve the optimization (pymanopt solver)

    Optional parameters
    -------------------
    :param pre_processing_manifold: a function that pre-process the data on the manifold before the optimization.
        Typically, this can be used to transform vectors (required by the GP) to the corresponding matrices (required
        for matrix-manifold optimization)
    :param post_processing_manifold: a function that post-process the data on the manifold after the optimization.
        Typically, this can be used to transform matrices (required for matrix-manifold optimization) to vectors
        (required by the GP).
    :param lower_bounds: minimum values for each column of initial_conditions
    :param upper_bounds: maximum values for each column of initial_conditions
    :param inequality_constraints: inequality constraint or list of inequality constraints, satisfied if >= 0
    :param equality_constraints: equality constraint or list of equality constraints, satisfied if = 0
    :param approx_hessian: if True, the Hessian of the cost is approximated with finite differences of the gradient
    :param solver_init_conds: if True, the initialization is made inside the solver. This has to be True for
        population-based methods, e.g. PSO, Nelder mead.
    :param options: options for candidate generation

    Returns
    -------
    :return: 2-element tuple containing the set of generated candidates and the acquisition value for each t-batch.
    """

    # options = options or {}
    # x0 = initial_conditions.requires_grad_(True)
    x0 = initial_conditions

    # If necessary pre-process the points
    if pre_processing_manifold is not None:
        # TODO this won't work for product of manifolds
        x0 = pre_processing_manifold(torch.from_numpy(x0)).cpu().numpy()

    @pymanopt.function.PyTorch
    def cost(*x):
        # If necessary post-process x
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x)

        if post_processing_manifold is not None:
            x = post_processing_manifold(x)

        x = x[None]
        x = x.to(device)
        loss = -acquisition_function(x).sum()
        # fval = loss.item()
        return loss

    # Define precon for the problem to avoid numerical issues
    # This precon can be removed if the condition Hd==0 is checked before lines 451-5 in TrustRegions.
    def precon(x, d):
        if isinstance(d, list) or isinstance(d, tuple):
            if np.sum(np.concatenate(d)) == 0.:
                for di in d:
                    di += 1e-30
        elif np.sum(d) == 0.:
            d += 1e-30
        return d

    # Instanciate the problem on the manifold
    problem = Problem(manifold=manifold, cost=cost, verbosity=0, precon=precon)

    # For cases where the Hessian is hard/long to compute, we approximate it with finite differences of the gradient.
    # Typical cases: the Hessian can be hard to compute due to the 2nd derivative of the eigenvalue decomposition,
    # e.g. in the SPD affine-invariant distance.
    if approx_hessian:
        problem._hess = types.MethodType(get_hessianfd, problem)

    # Solve problem on the manifold for each of the initial conditions
    if isinstance(x0, list):
        nb_initial_conditions = len(x0)
        dim_x = np.concatenate(x0[0]).shape[0]
        candidates = torch.zeros((nb_initial_conditions, 1, dim_x), dtype=torch.float64)
    else:
        nb_initial_conditions = x0.shape[0]
        candidates = torch.zeros(((nb_initial_conditions, 1) + x0.shape[1:]), dtype=torch.float64)

    # TODO this does not handle the case where q!=1
    for i in range(nb_initial_conditions):
        # with torch.autograd.detect_anomaly():
        if not solver_init_conds:
            if equality_constraints is not None or inequality_constraints is not None:
                opt_x = solver.solve(problem, x=x0[i],
                                     eq_constraints=equality_constraints, ineq_constraints=inequality_constraints)
            else:
                opt_x = solver.solve(problem, x=x0[i])
        else:
            # In this case, the solver takes care of the initialization. Typically applied to population-based methods,
            # e.g. PSO, Nelder Mead, that initialize a random population.
            # It could be better to provide also a good and controllerd initialization in this case. However, if the
            # population is big enough, the random initialization of the solver should be sufficient.
            opt_x = solver.solve(problem)

        if isinstance(opt_x, list) or isinstance(opt_x, tuple):
            opt_x = np.concatenate(opt_x)

        candidates[i] = torch.tensor(opt_x[None])

    # If necessary post-process the candidates
    if post_processing_manifold is not None:
        candidates = post_processing_manifold(candidates)

    candidates = candidates.to(device)
    batch_acquisition = acquisition_function(candidates)
    return candidates, batch_acquisition


# This function is based on the botorch.optim.gen_batch_initial_conditions function.
def gen_batch_initial_conditions_manifold(
    acq_function: AcquisitionFunction,
    manifold: Manifold,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    sample_type: torch.dtype = torch.float64,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
    post_processing_manifold: Optional[Callable[[Tensor], Tensor]] = None,
) -> Union[list, np.ndarray]:
    """
    This function generates a batch of initial conditions for random-restart optimization

    Parameters
    ----------
    :param acq_function: the acquisition function to be optimized.
    :param manifold: the manifold in the optimization takes place (pymanopt manifold)
    :param bounds: a `2 x d` tensor of lower and upper bounds for each column of `X`
    :param q: number of candidates
    :param num_restarts: number of starting points for multistart acquisition function optimization
    :param raw_samples: number of samples for initialization
    :param sample_type: type of the generated samples for initialization

    Optional parameters
    -------------------
    :param options: options for candidate generation
    :param post_processing_manifold: a function that post-process the data on the manifold after the optimization.
        Typically, this can be used to transform matrices (required for matrix-manifold optimization) to vectors
        (required by the GP).

    Returns
    -------
    :return: a `num_restarts x q x d` tensor of initial conditions
    """

    options = options or {}
    seed: Optional[int] = options.get("seed")  # pyre-ignore
    batch_limit: Optional[int] = options.get("batch_limit")  # pyre-ignore
    batch_initial_arms: Tensor
    factor, max_factor = 1, 5
    init_kwargs = {}
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch
    if q is None:
        q = 1

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            # Generate random points on the manifold
            manifold_samples = [manifold.rand() for i in range(raw_samples * factor * q)]
            if not isinstance(manifold, Product):
                points = [torch.from_numpy(point)[None, None] for point in manifold_samples]
            else:
                points = [torch.from_numpy(np.concatenate(point))[None, None] for point in manifold_samples]

            # Final tensor of random points
            # X_rnd = torch.cat(points).to(sample_type)
            X_rnd = torch.cat(points).to(device)

            # If necessary post-process the points
            if post_processing_manifold is not None:
                X_rnd = post_processing_manifold(X_rnd)

            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]

                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(X_rnd[start_idx:end_idx])
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit

                Y_rnd = torch.cat(Y_rnd_list).to(X_rnd)

            batch_initial_conditions = init_func(X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs)

            # Post-process the initial conditions if we have a product of manifolds
            if isinstance(manifold, Product):
                initial_conditions = []
                for i in range(batch_initial_conditions.shape[0]):
                    for j in range(batch_initial_conditions.shape[1]):
                        idx = (X_rnd == batch_initial_conditions[i, j]).nonzero(as_tuple=True)
                        # TODO if we use q!=1, we must check the following line.
                        initial_conditions.append(manifold_samples[idx[0][0] + raw_samples * idx[1][0]])
                batch_initial_conditions = initial_conditions
            # Otherwise, squeeze and transform to numpy array
            else:
                batch_initial_conditions = torch.squeeze(batch_initial_conditions).cpu().detach().numpy()

            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions

            if factor < max_factor:
                factor += 1

    warnings.warn("Unable to find non-zero acquisition function values - initial conditions are being selected "
                  "randomly.", BadInitialCandidatesWarning,)
    return batch_initial_conditions
