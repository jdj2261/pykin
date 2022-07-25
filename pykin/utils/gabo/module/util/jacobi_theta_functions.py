"""
Original code is part of the MaternGaBO library.
Github repo : https://github.com/NoemieJaquier/MaternGaBO
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
"""

import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'


def jacobi_theta_function3(z, q, serie_nb_terms=200):
    function_value = torch.zeros_like(z).to(device)
    for n in range(1, 1+serie_nb_terms):
        function_value += torch.pow(q, n**2) * torch.cos(2*n*z)
    return 2 * function_value + 1.