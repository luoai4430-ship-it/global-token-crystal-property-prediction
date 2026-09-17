import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
from torch_geometric.nn.conv import MessagePassing
from .config import Config as model_config
from data.config import Config as data_config


def gaussian_intervals(n: int = 64, start: float = -2.2, end: float = 4.0, mean: float = 0.0, std: float = 1.0):
    """
    Generate n one-dimensional spacers so that their density obeys N(mean, std^2).
    :param n:
    :param start:
    :param end:
    :param mean:
    :param std:

    :return:
    """
    # Take n equiprobable positions on (start,end).
    start_prob = norm.cdf(mean + start * std, mean, std)
    end_prob = norm.cdf(mean + end * std, mean, std)
    probs = np.linspace(start_prob, end_prob, n + 1, endpoint=False)
    # Mapping to the real number axis via ppf (inverse CDF).
    points = norm.ppf(probs, loc=mean, scale=std)
    return points


class RBFExpansion(nn.Module):
    """
    Expand interatomic distances with radial basis functions.
    Args:
        vmin: min of input
        vmax: max of input
        bins: expanded dimensions
        equal_density_interval: If True, use equal-density interval. Otherwise, use equal-distance interval.
            Default: ``False``.
        lengthscale:
    Shape:
        - Input: [num_nodes, 1]
        - Output: [num_nodes, bins]
    """

    def __init__(
            self,
            vmin: float = 0.0,
            vmax: float = 4.0,
            bins: int = 64,
            equal_density_interval: bool = False,
            lengthscale: float = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))
        if equal_density_interval:
            points = torch.from_numpy(gaussian_intervals(bins)).to(torch.float32).cuda()
            self.register_buffer('centers', points[:-1])
            self.lengthsale = torch.diff(points)
            self.gamma = 1 / self.lengthsale
        else:
            if lengthscale is None:
                # SchNet-style
                # set lengthscales relative to granularity of RBF expansion
                self.lengthscale = np.diff(self.centers).mean()
                self.gamma = 1 / self.lengthscale
            else:
                self.lengthscale = lengthscale
                self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        with torch.no_grad():
            return torch.exp(-self.gamma * (distance - self.centers) ** 2)


class SFTConv(MessagePassing):
    def __init__(self, node_feature: int = 128, dist_dim: int = 64, sh_dim: int = 64,
                 use_sh: bool = True, use_dist: bool = True):
        super().__init__()
        self.aggr = 'sum'
        self.node_feature = node_feature
        self.dist_dim = dist_dim
        self.sh_dim = sh_dim
        self.bond_dim = 2 * self.node_feature + self.dist_dim
        self.hidden_dim = 2 * self.node_feature + self.sh_dim * int(use_sh) + self.dist_dim * int(use_dist)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, self.node_feature),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.bond_dim, 512),
            nn.SiLU(),
            nn.Linear(512, self.node_feature),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm1d(self.node_feature)
        # self.act_fun = nn.ReLU()
        self.act_fun = nn.SiLU()

    def forward(self, x, edge_index, edge_attr):
        # Start the messaging process
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out) + x
        out = self.act_fun(out)
        return out

    def message(self, x_i, x_j, edge_attr, index):
        # bond = torch.cat([x_i, x_j, dist], dim=-1)
        # mess = torch.cat([x_i, x_j, dist, sh], dim=-1)
        # return self.gate(bond) * self.mlp(mess)
        combined = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.gate(combined[:, :self.bond_dim]) * self.mlp(combined)
