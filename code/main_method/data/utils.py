import torch
import numpy as np
from scipy.special import sph_harm, sph_harm_y
from torch import Tensor


def vector2sph(xyz: torch.Tensor | np.ndarray) -> tuple[Tensor, Tensor]:
    '''
        Cartesian coordinates to spherical coordinates
    :param xyz: shape [n, 3]
    :return: tuple(theta: shape [n, 3], phi: shape [n, 3])
    '''
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2],
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    # Calculate the polar angle, theta (0 <= theta <= pi)
    theta = torch.acos(z / r)
    # Calculate the azimuth angle, phi (0 <= phi <= 2*pi)
    phi = torch.atan2(y, x) + torch.pi
    return theta, phi



def spherical_harmonics_basis(theta, phi, n_orders):
    """
    Compute spherical harmonics basis.
    source code:https://www.cnblogs.com/ghostcai/p/17162034.html
    Args:
        theta (Tensor): polar angle, in range [0, pi].
        phi (Tensor): azimuthal angle, in range [0, 2*pi].
        n_orders (int): number of orders.
    Returns:
        Tensor: spherical harmonics basis, of shape (n_samples, n_orders**2).
    """
    # ! attention: scipy sph_harm swap (theta, phi) & (order, degree)
    # compute spherical harmonics basis
    basis = []
    for order in range(n_orders):
        for degree in range(-order, order + 1):
            #y_lm = sph_harm(abs(degree), order, phi, theta) # tensor -> tensor
            y_lm = sph_harm_y(order, abs(degree), theta, phi) # tensor -> ndarray
            if degree < 0:
                y_lm = np.sqrt(2) * (-1) ** degree * y_lm.imag
            elif degree > 0:
                y_lm = np.sqrt(2) * (-1) ** degree * y_lm.real
            #basis.append(y_lm.real)
            basis.append(torch.from_numpy(y_lm.real).type(torch.float32))
            #assert y_lm.real.dtype == torch.float32,f'dtype:{y_lm.real.dtype}'
    basis = torch.stack(basis, dim=1).float()  # [n_samples, n_orders**2]
    return basis
