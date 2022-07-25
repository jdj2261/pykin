"""
This file is part of the MaternGaBO library.
Authors: Noemie Jaquier and Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

from __future__ import division

from pymanopt.manifolds.sphere import Sphere
from pymanopt.manifolds.product import Product


class Torus(Product):

    def __init__(self, dimension):

        self.manifolds = [Sphere(2)] * dimension
        super(Torus, self).__init__(self.manifolds)
