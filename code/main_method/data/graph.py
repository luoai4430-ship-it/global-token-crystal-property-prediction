import torch
import torch_geometric
from torch_geometric.data import Data
import jarvis
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata

from data.utils import vector2sph, spherical_harmonics_basis
from data.config import Config


def crystalNN(structure: dict) -> tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
    """
        Create a multi-edge graph by CrystalNN
        Article title: Benchmarking Coordination Number Predicti on Algorithmson Inorganic Crystal Structures
        https://pubs.acs.org/doi/10.1021/acs.inorgchem.0c02996
    :param structure:
    :return:
    """
    pass


def atoms2graph(atoms: dict) -> torch_geometric.data.Data:
    """
        Create a multi-edge graph by KNN method
    :param atoms: jarvis
    :return:
    """
    structure = Atoms.from_dict(atoms)
    if not Config.use_crystalNN:
        edges = nearest_neighbor_edges(atoms=structure, cutoff=Config.cutoff, max_neighbors=Config.max_neighbors)
        u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)
    else:
        u, v, r = crystalNN(structure)

    dist = torch.norm(r, dim=1)
    data = Data(edge_index=torch.stack([u, v]).long(), edge_attr=dist)
    data.x = torch.tensor(structure.atomic_numbers, dtype=torch.long)

    if Config.use_sh:
        theta, phi = vector2sph(r)
        sh = spherical_harmonics_basis(theta, phi, Config.spherical_harmonics_l) * torch.sin(theta).reshape(-1, 1)
        data.sh = sh
    return data
