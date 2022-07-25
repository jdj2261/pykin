"""
Original code is part of the MaternGaBO library.
Github repo : https://github.com/NoemieJaquier/MaternGaBO
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
"""

import numpy as np


def get_hessianfd(self, x, a):
    """
    Compute an approximation of the Hessian w/ finite differences of the gradient.
    This function is based on the manopt function getHessianFD
    (https://www.manopt.org/reference/manopt/core/getHessianFD.html)
    and on the following paper:
    "Riemannian Trust Regions with Finite-Difference Hessian Approximations are Globally Convergent", N. Boumal, GSI'15.

    Parameters
    ----------
    :param self: problem of pymanopt
    :param x: base point to compute the Hessian
    :param a: direction where to compute the Hessian

    Returns
    -------
    :return: approximate Hessian
    """
    # Step size
    norm_a = self.manifold.norm(x, a)

    # Compute the gradient at the current point
    grad = self.grad(x)

    # Check that the step is not too small
    if norm_a < 1e-15:
        return np.zeros(grad.shape)

    # Parameter: how far do we look?
    epsilon = 2**(-14)

    c = epsilon/norm_a

    # Compute a point a little further along a and the gradient there.
    x1 = self.manifold.retr(x, c*a)

    grad1 = self.grad(x1)

    # Transport grad1 from x1 to x
    grad1 = self.manifold.transp(x1, x, grad1)

    # Return the finite difference of them
    if type(x) in (list, tuple) or issubclass(type(x), (list, tuple)):
        # Handle the case where x is a list or a tuple (typically for products of manifolds
        for k in range(len(x)):
            grad1[k] /= c
            grad[k] /= c
        finite_difference_grad = grad1 - grad
    else:
        finite_difference_grad = grad1/c - grad/c

    return finite_difference_grad
