import glob
import json
import math
import os
import pickle
import re
import time
from pathlib import Path
import random
import os.path as osp
import numpy as np
import torch

import pandas as pd
from typing import Literal

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms
from jarvis.core.specie import get_node_attributes

from pandarallel import pandarallel
from tqdm import tqdm

from data.config import *

from data.graph import atoms2graph


def _initialize_pandarallel() -> None:
    default_workers = min(8, os.cpu_count() or 1)
    workers = int(os.environ.get("SHFMAT_GRAPH_WORKERS", default_workers))
    if workers <= 0:
        raise ValueError("SHFMAT_GRAPH_WORKERS must be positive")
    pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=workers)
    print(f"Graph construction workers: {workers}")


def load_from_cif_csv(cif_dir: str):
    """
    从 cif_dir 下的 id_prop.csv 与 {id}.cif 加载数据，返回与原先 jdata 格式兼容的列表。
    - 固定读取 cif_dir/id_prop.csv（无表头，第 0 列 id、第 1 列 target）
    - 仅当 Config.target 为 MPTarget.ShearModuli 或 MPTarget.BulkModuli 时对 target 做 log10；JARVIS 的 bulk_modulus_kv、shear_modulus_gv 使用 GPa 原始值，不取对数
    - 对每个 id 读取 cif_dir/{id}.cif：优先 Jarvis Atoms.from_cif(..., use_cif2cell=False)，失败则 pymatgen Structure.from_file + pmg_to_atoms
    - 返回列表，每项含 atoms（dict）、id/jid（由 Config.dataset_name 决定）、Config.target.value
    """
    csv_path = osp.join(cif_dir, 'id_prop.csv')
    df = pd.read_csv(csv_path, header=None)
    id_col = df.iloc[:, 0]
    target_col = df.iloc[:, 1].astype(float)
    apply_log10 = Config.target in (MPTarget.ShearModuli, MPTarget.BulkModuli)
    id_tag = 'jid' if Config.dataset_name == CrystalDataset.Jarvis else 'id'
    out = []
    for idx in range(len(df)):
        sid = id_col.iloc[idx]
        target_val = target_col.iloc[idx]
        if apply_log10:
            target_val = math.log10(max(float(target_val), 1e-30))
        cif_path = osp.join(cif_dir, f'{str(sid).strip()}.cif')
        if not osp.isfile(cif_path):
            raise FileNotFoundError(f'CIF 文件不存在: {cif_path}')
        atoms = None
        try:
            atoms = Atoms.from_cif(cif_path, use_cif2cell=False)
        except Exception:
            try:
                from pymatgen.core.structure import Structure
                from jarvis.core.atoms import pmg_to_atoms
                struct = Structure.from_file(cif_path)
                atoms = pmg_to_atoms(struct)
            except Exception as e:
                raise RuntimeError(f'无法从 CIF 加载结构 {cif_path}: {e}') from e
        atoms_dict = atoms.to_dict() if hasattr(atoms, 'to_dict') else atoms
        out.append({
            'atoms': atoms_dict,
            id_tag: sid,
            Config.target.value: target_val,
        })
    return out


def _extract_embedding_tensor(v):
    """
    从 value 中取出向量：可为 tensor/ndarray，或 dict 中 'embedding'/'vec'/'features' 等 key。
    .pt 格式参照 MatBERT/MatbertToPt.py：每项为 { "embedding": tensor(768), "composition", "prop", "file_name" }。
    """
    if v is None:
        return None
    if isinstance(v, (torch.Tensor, np.ndarray)):
        return v
    if isinstance(v, dict):
        for key in ('embedding', 'vec', 'features', 'emb', 'hidden'):
            if key in v and v[key] is not None:
                return v[key]
        for k, x in v.items():
            if isinstance(x, (torch.Tensor, np.ndarray)):
                return x
    return None


def _load_text_embeddings_from_dir(dir_path: str, device='cpu'):
    """
    从目录加载多个 .pt 文件并合并为 id -> embedding 映射。
    用于 Bulk/Shear：train/val/test 各一个 .pt，合并后供 lookup。
    """
    import glob
    merged = {}
    for f in glob.glob(osp.join(dir_path, '*_MatBert_embeddings.pt')) or glob.glob(osp.join(dir_path, '*.pt')):
        obj = torch.load(f, map_location=device, weights_only=False)
        if isinstance(obj, dict):
            merged.update(obj)
    if not merged:
        raise ValueError(f"No .pt files found in {dir_path}")
    sample = next(iter(merged.values()))
    raw = _extract_embedding_tensor(sample)
    if raw is None:
        raise ValueError(f"Dict values must contain embedding tensor")
    dim = raw.numel() if raw.dim() == 0 else raw.size(-1)
    def _to_vec(t):
        if t is None:
            return torch.zeros(dim, device=device, dtype=torch.float32)
        t = _extract_embedding_tensor(t)
        if t is None:
            return torch.zeros(dim, device=device, dtype=torch.float32)
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float().to(device)
        else:
            t = t.clone().float().to(device)
        return t.reshape(-1)
    def lookup(cid):
        v = merged.get(cid)
        if v is None:
            v = merged.get(str(cid))
        return _to_vec(v)
    return lookup


def _load_text_embeddings(path: str, device='cpu'):
    """
    加载文本嵌入 .pt 文件，返回 id -> tensor(768) 的 lookup。

    .pt 格式参照 MatBERT/MatbertToPt.py：dict { id: { "embedding": tensor(768), "composition", "prop", "file_name" } }，
    由 description.csv（Id, Composition, prop, Description, File_Name）经 MatBERT 得到，保存为 CSV 同目录下 MatBert_embeddings.pt。

    也支持：
    (1) dict { id: { "embedding": tensor[768], ... } } 任意含 embedding 的 dict
    (2) dict { id: tensor[768] } 直接存向量
    (3) (ids_list, tensor Nx768)

    缺失 id 时返回零向量。lookup 时会尝试 cid 与 str(cid)。
    """
    if osp.isdir(path):
        return _load_text_embeddings_from_dir(path, device)
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        sample = next(iter(obj.values())) if obj else None
        if sample is None:
            raise ValueError(f"Dict in {path} is empty or has None values")
        raw = _extract_embedding_tensor(sample)
        if raw is None:
            raise ValueError(f"Dict values in {path} must be tensor/ndarray or dict containing one, got {type(sample)}")
        if isinstance(raw, torch.Tensor):
            dim = raw.numel() if raw.dim() == 0 else raw.size(-1)
        else:
            dim = np.ravel(raw).shape[0]
        def _to_vec(t):
            if t is None:
                return torch.zeros(dim, device=device, dtype=torch.float32)
            t = _extract_embedding_tensor(t)
            if t is None:
                return torch.zeros(dim, device=device, dtype=torch.float32)
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t).float().to(device)
            else:
                t = t.clone().float().to(device)
            return t.reshape(-1)
        def lookup(cid):
            v = obj.get(cid)
            if v is None:
                v = obj.get(str(cid))
            return _to_vec(v)
        return lookup
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        ids_list, tensor = obj[0], obj[1]
        id2idx = {str(k): i for i, k in enumerate(ids_list)}
        tensor = tensor.float().to(device)
        dim = tensor.size(-1)
        def lookup(cid):
            i = id2idx.get(str(cid))
            return tensor[i].clone() if i is not None else torch.zeros(dim, device=device, dtype=torch.float32)
        return lookup
    raise ValueError(f"Unsupported text embedding format in {path}: type {type(obj)}")


class _CrysAtomVec:
    """
        From CrysAtom: Distributed Representation of Atoms for Crystal Property Prediction
        https://github.com/shrimonmuke0202/CrysAtom
    """
    def __init__(self):
        with open(osp.join(str(Path(__file__).parent), 'CrysAtom_vector.pkl'), 'rb') as f:
            self.data = pickle.load(f)
        self.all_features = torch.zeros(size=(95, 200), dtype=torch.float)
        for k, v in self.data.items():
            self.all_features[k] = v.squeeze()

    def get(self, atomic_number: int | torch.Tensor) -> torch.Tensor:
        """
        :param atomic_number:
        :return: tensor shape[95,200]
        """
        if isinstance(atomic_number, torch.Tensor):
            atomic_number = int(atomic_number)
        return self.data[atomic_number]

    def getAllFeatures(self) -> torch.Tensor:
        """
        :return: tensor shape[95,200]
        """
        return self.all_features


class GraphDataset(InMemoryDataset):
    """
        The Dataset will further process the edge features of the node features on the basis of rawdata,
        and the processed data will be cached in the ./Dataset/processed path
    """
    def __init__(self,
                 rawdata,
                 split_type: Literal['train', 'valid', 'test'],
                 root=r'Dataset',
                 mean: float = 0.0,
                 std: float = 1.0,
                 ):
        self.rawdata = rawdata
        self.root = root
        self.split_type = split_type
        self.atom_features = Config.atom_features
        multimodal_suffix = '-multimodal' if Config.use_multimodal else ''
        if Config.use_multimodal:
            text_emb_path = getattr(Config, 'text_emb_path', '') or ''
            text_emb_stem = Path(text_emb_path).stem if text_emb_path else 'default_text'
            text_emb_stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', text_emb_stem)[:80]
            multimodal_suffix += f'-{text_emb_stem}'
        self.whole_name = Config.getDatasetWholeName() + f'-{self.split_type}-{Config.atom_features}{multimodal_suffix}'
        self.normalize = Config.normalize
        self.mean = mean
        self.std = std
        with open(osp.join(str(Path(__file__).parent), 'elements.json'), 'rt') as f:
            self.element_dict = json.load(f)  # key: str of atomic number,value: str of symbol

        super().__init__(self.root)
        del self.rawdata
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> str:
        return self.whole_name + '.pt'

    def process(self) -> None:
        match self.atom_features:
            case AtomFeatureType.AtomicNumber:
                with tqdm(self.rawdata, desc=f'processing {self.split_type} data') as t:
                    for data in t:
                        # To adapt to the embedding in the model, use this as the index and adjust the range to [0,93]
                        data.x = data.x - 1
                        if len(data.x.shape) == 1:
                            data.x = data.x.reshape(-1, 1)
                        data.x = data.x.reshape(-1)

            case AtomFeatureType.CGCNN | AtomFeatureType.CrysAtomVec:
                if self.atom_features == AtomFeatureType.CGCNN:
                    features = np.zeros(shape=(1 + 94, 92))
                    for z, symbol in self.element_dict.items():
                        features[int(z)] = get_node_attributes(symbol, self.atom_features.value)
                    features = torch.tensor(features, dtype=torch.float)
                elif self.atom_features == AtomFeatureType.CrysAtomVec:
                    features = _CrysAtomVec().getAllFeatures()

                with tqdm(self.rawdata, desc=f'processing {self.split_type} data') as t:
                    for data in t:
                        data.x = features[data.x]
                        if len(data.x.shape) == 1:  # If there is only one atom in the crystal unit cell
                            data.x = data.x.unsqueeze(0)
                        if self.normalize:
                            data.y = (data.y - self.mean) / self.std

            case _:
                raise ValueError(f'Unknown atom_features: {self.atom_features}')

        data, slices = self.collate(self.rawdata)
        torch.save((data, slices), self.processed_paths[0])


def get_dataloader():
    """
        Generate a dataset and dataloader for a single task based on data.config.config,
        and save the cache file in the ./dataset path during the generation process,
        which can be deleted if it is no longer trained or tested
    :return: train_loader, valid_loader, test_loader, mean, std
    """
    if Config.target in JarvisTarget:
        Config.dataset_name = CrystalDataset.Jarvis
    else:
        Config.dataset_name = CrystalDataset.MP

    dataset_target_pair = {CrystalDataset.Jarvis: JarvisTarget, CrystalDataset.MP: MPTarget}
    if Config.target not in dataset_target_pair[Config.dataset_name]:
        raise ValueError(f'The dataset:{Config.dataset_name} does not match the target:{Config.target}!')
    del dataset_target_pair[Config.dataset_name]

    if Config.use_cif_csv:
        if not getattr(Config, 'cif_dir', '') or not osp.isdir(Config.cif_dir):
            raise ValueError(f'use_cif_csv=True 时需指定有效 cif_dir 目录: {getattr(Config, "cif_dir", "")}')
        if not osp.isfile(osp.join(Config.cif_dir, 'id_prop.csv')):
            raise FileNotFoundError(f'cif_dir 下缺少 id_prop.csv: {osp.join(Config.cif_dir, "id_prop.csv")}')

    if Config.target in [MPTarget.BulkModuli, MPTarget.ShearModuli] and not Config.use_cif_csv:
        if not osp.isfile(Config.raw_data_cache_path()):
            pkl_dir = getattr(Config, 'bulk_shear_pkl_dir', '') or ''
            if pkl_dir and osp.isdir(pkl_dir):
                train_files = glob.glob(osp.join(pkl_dir, '*train*.pkl'))
                val_files = glob.glob(osp.join(pkl_dir, '*val*.pkl'))
                test_files = glob.glob(osp.join(pkl_dir, '*test*.pkl'))
                if not (train_files and val_files and test_files):
                    raise FileNotFoundError(f'bulk_shear_pkl_dir 需包含 *train*.pkl, *val*.pkl, *test*.pkl: {pkl_dir}')
                with open(train_files[0], 'rb') as f:
                    data_train = pickle.load(f)
                with open(val_files[0], 'rb') as f:
                    data_valid = pickle.load(f)
                with open(test_files[0], 'rb') as f:
                    data_test = pickle.load(f)
                print(f'[Bulk/Shear] Loaded pkl from {pkl_dir}')
            else:
                filename_dict = {MPTarget.BulkModuli: 'bulk_megnet', MPTarget.ShearModuli: 'shear_megnet'}
                with open(osp.join('data', 'MPElasticModulus', filename_dict[Config.target] + '_train.pkl'), 'rb') as f:
                    data_train = pickle.load(f)
                with open(osp.join('data', 'MPElasticModulus', filename_dict[Config.target] + '_val.pkl'), 'rb') as f:
                    data_valid = pickle.load(f)
                with open(osp.join('data', 'MPElasticModulus', filename_dict[Config.target] + '_test.pkl'), 'rb') as f:
                    data_test = pickle.load(f)
            torch.set_printoptions(precision=10)
            text_emb_path_for_bulk_shear = getattr(Config, 'bulk_shear_text_emb', '') or Config.text_emb_path
            id_to_emb = None
            if Config.use_multimodal and text_emb_path_for_bulk_shear:
                if osp.isdir(text_emb_path_for_bulk_shear) or osp.isfile(text_emb_path_for_bulk_shear):
                    id_to_emb = _load_text_embeddings(text_emb_path_for_bulk_shear)
                    print(f'[multimodal] Loaded Bulk/Shear text embeddings from {text_emb_path_for_bulk_shear}')
                else:
                    print(f'[multimodal] WARNING: text path not found: {text_emb_path_for_bulk_shear}')
            all_data = []
            for original_data in [data_train, data_valid, data_test]:
                graph_data = [data['atoms'] for data in original_data]
                df_graph = pd.DataFrame(graph_data)
                if Config.use_parallel_graph:
                    _initialize_pandarallel()
                    graph_data = df_graph.parallel_apply(atoms2graph, axis=1)
                else:
                    tqdm.pandas()
                    graph_data = df_graph.progress_apply(atoms2graph, axis=1)
                graph_data = graph_data.values.tolist()
                if Config.target == MPTarget.BulkModuli:
                    for idx, data in enumerate(graph_data):
                        data.y = original_data[idx][Config.target.value].float()
                elif Config.target == MPTarget.ShearModuli:
                    for idx, data in enumerate(graph_data):
                        data.y = torch.tensor(original_data[idx][Config.target.value], dtype=torch.float)
                if Config.use_multimodal:
                    for idx, data in enumerate(graph_data):
                        rec = original_data[idx]
                        cid = rec.get('id') or rec.get('material_id') or rec.get('mp_id')
                        data.cid = cid
                        data.text_emb = id_to_emb(cid) if id_to_emb is not None else torch.zeros(768)
                all_data.append(graph_data)
            torch.save(all_data, Config.raw_data_cache_path())
            data_train, data_valid, data_test = all_data
        else:
            data_train, data_valid, data_test = torch.load(Config.raw_data_cache_path())
            text_emb_path_for_bulk_shear = getattr(Config, 'bulk_shear_text_emb', '') or Config.text_emb_path
            if Config.use_multimodal and text_emb_path_for_bulk_shear and (osp.isdir(text_emb_path_for_bulk_shear) or osp.isfile(text_emb_path_for_bulk_shear)):
                id_to_emb = _load_text_embeddings(text_emb_path_for_bulk_shear)
                for split in [data_train, data_valid, data_test]:
                    for data in split:
                        if hasattr(data, 'cid') and data.cid is not None:
                            data.text_emb = id_to_emb(data.cid)
                total_n = len(data_train) + len(data_valid) + len(data_test)
                print(f'[multimodal] Re-loaded Bulk/Shear text embeddings from {text_emb_path_for_bulk_shear} ({total_n} samples)')

    else:
        if not osp.isfile(Config.raw_data_cache_path()):
            if Config.use_cif_csv:
                data = load_from_cif_csv(Config.cif_dir)
                if not data:
                    raise ValueError(
                        f'load_from_cif_csv 未得到任何样本，请检查 {Config.cif_dir}/id_prop.csv 与 {Config.cif_dir}/*.cif'
                    )
            else:
                data = jdata(Config.dataset_name.value)
            clean_data = []
            targets = []
            data_id = []
            id_tag = 'jid' if Config.dataset_name == CrystalDataset.Jarvis else 'id'
            for i in data:
                if isinstance(i[Config.target.value], list):
                    clean_data.append(i['atoms'])
                    targets.append(torch.tensor(i[Config.target.value]))
                    data_id.append(i[id_tag])

                elif (
                        i[Config.target.value] is not None
                        and i[Config.target.value] != "na"
                        and not math.isnan(i[Config.target.value])
                ):
                    clean_data.append(i['atoms'])
                    targets.append(i[Config.target.value])
                    data_id.append(i[id_tag])

            # clean_data = clean_data[:1000]
            torch.set_printoptions(precision=10)
            start_time = time.time()
            df_clean = pd.DataFrame(clean_data)
            if Config.use_parallel_graph:
                # progress_bar=True 时 pandarallel 常在 100% 后卡在汇总结果阶段，故关闭进度条并打印提示
                _initialize_pandarallel()
                print('Building graphs (parallel)...')
                clean_data = df_clean.parallel_apply(atoms2graph, axis=1)
                print('Merging results from workers...')
            else:
                tqdm.pandas()
                print('Building graphs (sequential, use_parallel_graph=False to avoid hang)...')
                clean_data = df_clean.progress_apply(atoms2graph, axis=1)
            end_time = time.time()
            print(f'\nTime consumption for constructing graph data: {end_time - start_time}s')

            clean_data = clean_data.values.tolist()
            id_to_emb = None
            # 多模态时 .pt 格式参照 MatbertToPt.py；CIF 模式下未指定 text_emb_path 则尝试 cif_dir/MatBert_embeddings.pt
            text_emb_path_resolved = Config.text_emb_path or (
                osp.join(Config.cif_dir, 'MatBert_embeddings.pt') if Config.use_cif_csv and Config.cif_dir else ''
            )
            if Config.use_multimodal and text_emb_path_resolved and osp.isfile(text_emb_path_resolved):
                id_to_emb = _load_text_embeddings(text_emb_path_resolved)
                print(f'[multimodal] Loaded text embeddings from {text_emb_path_resolved}')
            elif Config.use_multimodal and Config.text_emb_path:
                print(f'[multimodal] WARNING: text_emb_path not found, using zero text embeddings: {Config.text_emb_path}')
            for idx, data in enumerate(clean_data):
                data.y = torch.tensor(targets[idx], dtype=torch.float)
                data.cid = data_id[idx]
                if Config.use_multimodal:
                    data.text_emb = id_to_emb(data_id[idx]) if id_to_emb is not None else torch.zeros(768)

            torch.save(clean_data, Config.raw_data_cache_path())
        else:
            clean_data = torch.load(Config.raw_data_cache_path())
            # 从缓存加载后，若多模态且当前有文本嵌入文件，用当前文件重新挂载 text_emb（.pt 格式参照 MatbertToPt.py）
            text_emb_path_resolved = Config.text_emb_path or (
                osp.join(Config.cif_dir, 'MatBert_embeddings.pt') if Config.use_cif_csv and Config.cif_dir else ''
            )
            if Config.use_multimodal and text_emb_path_resolved and osp.isfile(text_emb_path_resolved):
                id_to_emb = _load_text_embeddings(text_emb_path_resolved)
                for data in clean_data:
                    if hasattr(data, 'cid'):
                        data.text_emb = id_to_emb(data.cid)
                print(f'[multimodal] Re-loaded text embeddings from {text_emb_path_resolved} (cache had {len(clean_data)} samples)')

        random.seed(Config.split_seed)
        random.shuffle(clean_data)
        num_train = 0
        num_valid = 0
        if Config.dataset_name == CrystalDataset.Jarvis:
            total_len = len(clean_data)
            num_train, num_valid = int(Config.train_ratio * total_len), int(Config.valid_ratio * total_len)
        elif Config.dataset_name == CrystalDataset.MP:
            total_len = len(clean_data)
            # 若数据总量不足 num_train+num_valid，则按比例划分，避免 valid/test 为空
            if total_len < Config.num_train + Config.num_valid:
                num_train = int(Config.train_ratio * total_len)
                num_valid = int(Config.valid_ratio * total_len)
            else:
                num_train, num_valid = Config.num_train, Config.num_valid
        else:
            raise ValueError(f'Unknown dataset_name: {Config.dataset_name}')

        data_train = clean_data[:num_train]
        data_valid = clean_data[num_train:num_train + num_valid]
        data_test = clean_data[num_train + num_valid:]

    # 多模态校验：确保至少有一部分样本带有非零文本嵌入，否则报错
    if Config.use_multimodal:
        all_data = data_train + data_valid + data_test
        n_with_text = 0
        for d in all_data:
            if getattr(d, 'text_emb', None) is not None and isinstance(d.text_emb, torch.Tensor):
                if d.text_emb.numel() > 0 and d.text_emb.abs().sum().item() > 1e-9:
                    n_with_text += 1
        total = len(all_data)
        print(f'[multimodal] {n_with_text} / {total} samples have non-zero text embeddings')
        if n_with_text == 0 and Config.text_emb_path:
            raise RuntimeError(
                'Multimodal is enabled but no sample has non-zero text embedding. '
                'Check text_emb_path and that .pt keys (e.g. mp_id) match dataset id.'
            )
        if n_with_text > 0:
            print('[multimodal] Text modality will be used (contrastive + property).')

    total_len = len(data_train) + len(data_valid) + len(data_test)
    print(f'dataset split: total={total_len}, train={len(data_train)}, valid={len(data_valid)}, test={len(data_test)}')

    def _safe_min_max(data_list, fallback_min=None, fallback_max=None):
        if not data_list:
            return fallback_min, fallback_max
        return min(data_list, key=lambda d: float(d.y)).y.item(), max(data_list, key=lambda d: float(d.y)).y.item()

    train_min, train_max = _safe_min_max(data_train)
    valid_min, valid_max = _safe_min_max(data_valid, train_min, train_max)
    test_min, test_max = _safe_min_max(data_test, train_min, train_max)
    print(f'train range:({train_min},{train_max})')
    print(f'valid range:({valid_min},{valid_max})')
    print(f'test range:({test_min},{test_max})')

    if Config.normalize:
        train_targets = torch.tensor([data.y for data in data_train])
        mean = torch.mean(train_targets).item()
        std = torch.std(train_targets).item()
        print(f'mean:{mean}')
        print(f'std:{std}')
    else:
        mean = 0.0
        std = 1.0
    train_dataset = GraphDataset(data_train, 'train', mean=mean, std=std)
    valid_dataset = GraphDataset(data_valid, 'valid', mean=mean, std=std)
    test_dataset = GraphDataset(data_test, 'test', mean=mean, std=std)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers,
                              pin_memory=Config.pin_memory, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False,
                              num_workers=Config.num_workers, pin_memory=Config.pin_memory, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=Config.num_workers,
                             pin_memory=Config.pin_memory, drop_last=False)

    return train_loader, valid_loader, test_loader, mean, std
