import enum
import os
import os.path as osp
from typing import Union


try:
    _StrEnum = enum.StrEnum
except AttributeError:
    class _StrEnum(str, enum.Enum):
        """Python 3.10-compatible replacement for enum.StrEnum."""


def _name_string(cls):
    cls.__str__ = lambda self: self.name
    return cls


@_name_string
class CrystalDataset(_StrEnum):
    Jarvis = "dft_3d"
    MP = "megnet"


@_name_string
class JarvisTarget(_StrEnum):
    FormationEnergy = "formation_energy_peratom"
    Bandgap_OPT = "optb88vdw_bandgap"
    Bandgap_MBJ = "mbj_bandgap"
    TotalEnergy = "optb88vdw_total_energy"
    BulkModulusKv = "bulk_modulus_kv"
    ShearModulusGv = "shear_modulus_gv"
    Ehull = "ehull"


@_name_string
class MPTarget(_StrEnum):
    FormationEnergy = "e_form"
    Bandgap = "gap pbe"
    BulkModuli = "bulk modulus"
    ShearModuli = "shear modulus"


@_name_string
class AtomFeatureType(_StrEnum):
    AtomicNumber = "atomic_number"
    CGCNN = "cgcnn"
    CrysAtomVec = "crys_atom"


class Config:
    dataset_name: CrystalDataset = CrystalDataset.Jarvis
    target: Union[JarvisTarget, MPTarget] = JarvisTarget.FormationEnergy
    atom_features: AtomFeatureType = AtomFeatureType.CGCNN

    num_train: int = 60000
    num_valid: int = 5000
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 123
    normalize: bool = True

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    cutoff: float = 4.0
    max_neighbors: int = 16
    use_sh: bool = True
    spherical_harmonics_l: int = 4
    use_crystalNN: bool = False
    use_parallel_graph: bool = True

    use_cif_csv: bool = False
    cif_dir: str = ""
    use_multimodal: bool = False
    text_emb_path: str = ""
    bulk_shear_pkl_dir: str = ""
    bulk_shear_text_emb: str = ""

    @staticmethod
    def info() -> str:
        attrs = [
            "dataset_name", "target", "atom_features", "split_seed",
            "batch_size", "normalize", "pin_memory", "cutoff",
            "max_neighbors", "use_sh", "use_crystalNN",
            "spherical_harmonics_l", "use_cif_csv", "use_multimodal",
        ]
        lines = ["data_config:", "{"]
        lines.extend(f"    {attr}={getattr(Config, attr)}" for attr in attrs)
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def getAtomDim() -> int:
        return {
            AtomFeatureType.AtomicNumber: 1,
            AtomFeatureType.CGCNN: 92,
            AtomFeatureType.CrysAtomVec: 200,
        }[Config.atom_features]

    @staticmethod
    def getTargetName() -> str:
        return f"{Config.dataset_name.name}-{Config.target.name}"

    @staticmethod
    def getDatasetWholeName() -> str:
        name = (
            f"{Config.dataset_name.name}-{Config.target.name}"
            f"-cutoff{Config.cutoff}-maxNeibNum{Config.max_neighbors}"
            f"-spheHarmL{Config.spherical_harmonics_l}"
            f"-splitSeed{Config.split_seed}"
        )
        if Config.use_crystalNN:
            name += "-withcrystalNN"
        if Config.use_cif_csv:
            name += "-CIF_CSV"
        return name

    @staticmethod
    def raw_data_cache_path() -> str:
        suffix = "-CIF_CSV" if Config.use_cif_csv else ""
        filename = (
            f"{Config.dataset_name.name}-{Config.target.name}"
            f"-cutoff{Config.cutoff}-maxNeibNum{Config.max_neighbors}"
            f"-spheHarmL{Config.spherical_harmonics_l}{suffix}-rawData.pt"
        )
        root = osp.join("Dataset", "raw")
        os.makedirs(root, exist_ok=True)
        return osp.join(root, filename)
