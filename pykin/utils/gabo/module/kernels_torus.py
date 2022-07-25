"""
Original code is part of the MaternGaBO library.
Github repo : https://github.com/NoemieJaquier/MaternGaBO
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
"""

import torch
import gpytorch
import numpy as np


from pykin.utils.gabo.module.util.sphere_utils_torch import sphere_distance_torch
from pykin.utils.gabo.module.util.jacobi_theta_functions import jacobi_theta_function3

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'

class TorusProductOfManifoldsRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on a torus by considering
    it as a product of circle manifolds.

    Attributes
    ----------
    self.dim, dimension of the torus manifold on which the data handled by the kernel are living
    self.torus_kernel, product of circle kernels

    Methods
    -------
    forward(point1_on_torus, point2_on_torus, diagonal_matrix_flag=False, **params):

    """
    def __init__(self, dim, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the torus manifold on which the data handled by the kernel are living

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(TorusProductOfManifoldsRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None,
                                                                              **kwargs)

        # Dimension of the torus
        self.dim = dim

        # Initialise the product of kernels
        kernels = [CircleRiemannianGaussianKernel(active_dims=torch.tensor(list(range(2*i, 2*i+2))))
                   for i in range(self.dim)]

        self.torus_kernel = gpytorch.kernels.ProductKernel(*kernels)

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a torus manifold by considering it
        as a product of circle manifolds

        Parameters
        ----------
        :param x1: input points on the torus
        :param x2: input points on the torus

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # If the points are given as angles, transform them into coordinates on circles
        if x1.shape[-1] == self.dim:
            x1_circles = torch.zeros(list(x1.shape[:-1]) + [2 * self.dim], dtype=x1.dtype)
            x1_circles[..., ::2] = torch.cos(x1)
            x1_circles[..., 1::2] = torch.sin(x1)
        else:
            x1_circles = x1
        if x2.shape[-1] == self.dim:
            x2_circles = torch.zeros(list(x2.shape[:-1]) + [2 * self.dim], dtype=x2.dtype)
            x2_circles[..., ::2] = torch.cos(x2)
            x2_circles[..., 1::2] = torch.sin(x2)
        else:
            x2_circles = x2

        # Kernel
        kernel = self.torus_kernel.forward(x1_circles, x2_circles)
        return kernel

class CircleRiemannianGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the circle, i.e.,
    sphere manifold S¹.

    Attributes
    ----------
    self.serie_nb_terms, number of terms used to compute the Jacobi theta function of the kernel

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, serie_nb_terms=100,  **kwargs):
        """
        Initialisation.

        Parameters
        ----------

        Optional parameters
        -------------------
        :param serie_nb_terms: number of terms used to compute the summation formula of the kernel
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(CircleRiemannianGaussianKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

        # Number of term used to compute the jacobi theta function
        self.serie_nb_terms = serie_nb_terms

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a circle / sphere manifold S^1.

        Parameters
        ----------
        :param x1: input points on the circle
        :param x2: input points on the circle

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        scaled_distance = sphere_distance_torch(x1, x2, diag=diag)/(2*np.pi)

        # Compute kernel equal to jacobi theta function
        q_param = torch.exp(-2 * np.pi**2 * self.lengthscale**2)
        kernel = jacobi_theta_function3(np.pi * scaled_distance, q_param).to(device)

        # Normalizing term
        norm_factor = jacobi_theta_function3(torch.zeros((1, 1)).to(device), q_param).to(device)

        # Kernel
        return kernel / norm_factor