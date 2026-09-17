import logging
import os
import os.path as osp
import random
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from tqdm import tqdm
from functools import total_ordering
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_dataloader
from data.config import Config as data_config, AtomFeatureType
from data.config import CrystalDataset, JarvisTarget, MPTarget
from model.net import SFTGNN, SFTGNNMultimodal
from model.adaptive_graph_preserving_text_residual import (
    AdaptiveGraphPreservingTextResidualSHFMat,
)
from model.text_residual_shfmat import TextResidualSHFMat
from model.config import Config as model_config
from src.loss.infonce import infonce_loss


def _extract_llm_type() -> str:
    """
    从 text_emb_path 的 .pt 文件名中提取大预言模型类型。
    例如：
        .../MatBert_embeddings.pt   -> MatBert
        .../RoBERTa_embeddings.pt   -> RoBERTa
        .../T5_embeddings.pt        -> T5
        .../SciBERT_embeddings.pt   -> SciBERT
    如果未指定 text_emb_path 或无法解析，返回空字符串。
    """
    emb_path = getattr(data_config, 'text_emb_path', '') or getattr(model_config, 'text_emb_path', '') or ''
    if not emb_path:
        return ''
    pt_filename = osp.basename(emb_path)                  # 例如 "RoBERTa_embeddings.pt"
    name_no_ext = osp.splitext(pt_filename)[0]             # 例如 "RoBERTa_embeddings"
    # 去除常见后缀 "_embeddings" / "_Embeddings" 得到模型类型
    for suffix in ('_embeddings', '_Embeddings', '_emb', '_Emb'):
        if name_no_ext.endswith(suffix):
            return name_no_ext[:-len(suffix)]
    # 如果不符合约定格式，直接用去掉扩展名的文件名
    return name_no_ext


def create_run_dir() -> str:
    """
    为本次运行创建唯一的输出目录。
    多模态时：Pic-log/{model}_{target}_{llm_type}_{YYYYMMDD_HHMMSS}/
    非多模态：Pic-log/{model}_{target}_{YYYYMMDD_HHMMSS}/
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    llm_type = _extract_llm_type()
    tag = re.sub(r'[^A-Za-z0-9_.-]+', '-', getattr(model_config, 'run_name', '')).strip('-')
    if llm_type:
        run_name = f'{model_config.model_name}_{data_config.getTargetName()}_{llm_type}_{timestamp}'
    else:
        run_name = f'{model_config.model_name}_{data_config.getTargetName()}_{timestamp}'
    if tag:
        run_name = f'{run_name}_{tag}'
    run_dir = osp.join('Pic-log', run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def getLogger(run_dir: str = None) -> logging.Logger:
    """
        Set up logger
    :return:
    """
    logger = logging.getLogger(data_config.getTargetName())
    logger.setLevel(logging.INFO)
    # 清除旧 handlers，避免多次调用时重复
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # 总日志（保持兼容）
    log_root = r'./Pic-log'
    if not osp.exists(log_root):
        os.makedirs(log_root)
    log_file = osp.join(log_root, f'{model_config.model_name}_{data_config.getTargetName()}.log')
    file_handler = logging.FileHandler(filename=log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 本次运行的日志
    if run_dir:
        run_log_file = osp.join(run_dir, 'run.log')
        run_file_handler = logging.FileHandler(filename=run_log_file, mode='a')
        run_file_handler.setFormatter(formatter)
        logger.addHandler(run_file_handler)
    return logger


@total_ordering
class ModelPara:
    """
        Save the model parameters, as well as the epoch and evaluation metrics at that time.
        The .pth binary actually holds that object
    """

    def __init__(self, model_para, score: float, epoch: int):
        self.model_para = model_para
        self.score = score
        self.epoch = epoch

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return f'model epoch: {self.epoch:0>3}, Valid MAE: {self.score:.8f}'


class TopKModelSaver:
    def __init__(self, model_para, k: int, early_stop: bool = False, patience: int = 15):
        """
            The best k models are saved during training
        :param model_para: model.state_dict()
        :param k:  The number of the best models saved
        :param early_stop: Whether to use an early stop mechanism
        :param patience: When the early stop mechanism is enabled, if the model evaluation indicators do not have better parameters after consecutive iterations, the early stop will be triggered and the training will be stopped
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.top_k = [ModelPara(deepcopy(model_para), float('inf'), 0) for _ in range(k)]
        self.early_stop = early_stop
        self.patience = patience
        self.counter = 0

    def step(self, model, score: float, epoch: int):
        """
        called once per epoch
        :param score: current validation metrics
        :param model: current model
        :param epoch: current epoch number for file naming
        """
        # update k optimal models
        temp_para = ModelPara(deepcopy(model.state_dict()), score, epoch)
        if temp_para < self.top_k[-1]:
            self.top_k.append(temp_para)
            self.top_k.sort()
            self.top_k.pop()
        else:
            self.counter += 1

    @property
    def stop(self) -> bool:
        if self.early_stop and self.counter >= self.patience:
            return True
        else:
            return False

    def state_dict(self) -> dict:
        return self.__dict__

    def load_state_dict(self, state_dict: dict):
        for key, value in state_dict.items():
            setattr(self, key, value)

    def info(self) -> str:
        s = f'The best {self.k}:\n'
        for i in self.top_k:
            s += str(i) + '\n'
        return s.removesuffix('\n')


class TrainConfigManager:
    """
        If the hyperparameter settings for each task are different during training, you can use this type to make special modifications,
        __enter__ adjust the hyperparameters, __exit__ restore the adjusted hyperparameters to their default values
    """

    def __init__(self, target: Union[JarvisTarget, MPTarget, None]):
        if target in [*JarvisTarget, *MPTarget, None]:
            self.target = target
        else:
            raise ValueError(f"target needs to be an enum value of JarvisTarget or MPTarget, or None. target:{target}")

    def __enter__(self):
        if self.target in (JarvisTarget.Bandgap_MBJ, MPTarget.BulkModuli, MPTarget.ShearModuli):
            model_config.num_epoch = 100

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.target in (JarvisTarget.Bandgap_MBJ, MPTarget.BulkModuli, MPTarget.ShearModuli):
            model_config.num_epoch = 500


def train(model, logger, train_loader, valid_loader, mean, std, model_saver: TopKModelSaver, run_dir: str = None):
    begin_epoch = model_config.begin_epoch
    num_epoch = model_config.num_epoch

    train_dataset_len = len(train_loader)
    valid_dataset_len = len(valid_loader)

    is_adaptive = model_config.model_name == 'AdaptiveGraphPreservingTextResidualSHFMat'
    loss = nn.L1Loss().cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=model_config.lr, weight_decay=model_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=model_config.max_lr, epochs=num_epoch,
                                                    steps_per_epoch=len(train_loader), pct_start=0.3)

    parameter_dir = osp.join(run_dir, 'parameter') if run_dir else 'parameter'
    os.makedirs(parameter_dir, exist_ok=True)

    if begin_epoch != 0:
        logger.info(f'Continue training from epoch:{begin_epoch}')
        para_file_name = f'{model_config.model_name}_{data_config.getTargetName()}_epoch{begin_epoch}_para.pth'
        model.load_state_dict(torch.load(osp.join(parameter_dir, para_file_name)))
        checkpoint = torch.load(osp.join(parameter_dir, 'checkpoint.pt'))
        model_saver.load_state_dict(checkpoint['model_saver'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("Training...")

    loss_train_trace = []
    loss_valid_trace = []
    train_metrics_rows = []  # 收集每个 epoch 的 train_loss, valid_loss, train_mae, valid_mae
    tensorboard_path = osp.join(run_dir, 'tensorboard') if run_dir else './Pic-log/tensorboard'
    if not osp.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    start_time = time.time()
    for epoch in range(begin_epoch, num_epoch):
        train_mae = 0
        train_graph_mae = 0
        train_alpha = 0
        train_scaled_delta = 0
        loss_sum = 0
        model.train()
        with tqdm(train_loader, leave=False, desc=f'epoch{epoch}:train') as pbar:
            for batch in pbar:
                batch = batch.cuda()
                out = model(batch)
                if isinstance(out, dict):
                    property_pred = out['property_pred']
                    result_loss = loss(property_pred, batch.y)
                    if is_adaptive:
                        graph_loss = loss(out['pred_graph'], batch.y)
                        lambda_graph = getattr(model_config, 'adaptive_lambda_graph', 0.5)
                        lambda_delta = getattr(model_config, 'adaptive_lambda_delta', 0.001)
                        lambda_residual_fit = getattr(
                            model_config, 'adaptive_lambda_residual_fit', 0.0
                        )
                        residual_fit_loss = torch.zeros(
                            (), device=batch.y.device, dtype=batch.y.dtype
                        )
                        if lambda_residual_fit:
                            residual_target = batch.y - out['pred_graph'].detach()
                            alpha_safe = out['alpha'].detach().clamp_min(1e-3)
                            delta_target = residual_target / alpha_safe
                            residual_fit_loss = F.smooth_l1_loss(
                                out['delta_text'], delta_target
                            )
                        result_loss = (
                            result_loss
                            + lambda_graph * graph_loss
                            + lambda_delta * torch.mean(torch.abs(out['scaled_delta']))
                            + lambda_residual_fit * residual_fit_loss
                        )
                        train_graph_mae += torch.absolute(out['pred_graph'] - batch.y).mean()
                        train_alpha += out['alpha'].mean()
                        train_scaled_delta += torch.abs(out['scaled_delta']).mean()
                    if model_config.model_name == 'TextResidualSHFMat':
                        lambda_delta = getattr(model_config, 'text_residual_lambda_delta', 0.0)
                        if lambda_delta:
                            scaled_delta = out['alpha'] * out['delta_text']
                            result_loss = result_loss + lambda_delta * torch.mean(scaled_delta ** 2)
                    if model_config.model_name == 'SFTGNNMultimodal' and getattr(model_config, 'use_contrastive', False) and 'embeddings' in out:
                        emb_text = out['embeddings']['text']
                        emb_crystal = out['embeddings']['crystal']
                        result_loss = result_loss + getattr(model_config, 'contrastive_loss_weight', 0.1) * infonce_loss(
                            emb_text, emb_crystal, getattr(model_config, 'contrastive_temperature', 0.07)
                        )
                else:
                    property_pred = out
                    result_loss = loss(property_pred, batch.y)
                loss_sum += result_loss.detach()
                train_mae += torch.absolute(property_pred - batch.y).mean()
                optimizer.zero_grad()
                result_loss.backward()
                optimizer.step()
                scheduler.step()

        train_mae = (train_mae.item() / train_dataset_len) * std
        if is_adaptive:
            train_graph_mae = (train_graph_mae.item() / train_dataset_len) * std
            train_alpha = train_alpha.item() / train_dataset_len
            train_scaled_delta = (train_scaled_delta.item() / train_dataset_len) * std
        loss_train_trace.append(loss_sum.item() / len(train_loader))

        valid_mae = 0
        valid_graph_mae = 0
        valid_alpha = 0
        valid_scaled_delta = 0
        loss_sum = 0
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader, leave=False, desc=f'epoch{epoch}:valid') as pbar:
                for batch in pbar:
                    batch = batch.cuda()
                    out = model(batch)
                    property_pred = out['property_pred'] if isinstance(out, dict) else out
                    result_loss = loss(property_pred, batch.y)
                    loss_sum += result_loss
                    valid_mae += torch.absolute(property_pred - batch.y).mean()
                    if is_adaptive:
                        valid_graph_mae += torch.absolute(out['pred_graph'] - batch.y).mean()
                        valid_alpha += out['alpha'].mean()
                        valid_scaled_delta += torch.abs(out['scaled_delta']).mean()

        valid_mae = (valid_mae.item() / valid_dataset_len) * std
        if is_adaptive:
            valid_graph_mae = (valid_graph_mae.item() / valid_dataset_len) * std
            valid_alpha = valid_alpha.item() / valid_dataset_len
            valid_scaled_delta = (valid_scaled_delta.item() / valid_dataset_len) * std
        loss_valid_trace.append(loss_sum.item() / len(valid_loader))
        train_loss = loss_train_trace[-1]
        valid_loss = loss_valid_trace[-1]
        train_metrics_rows.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_mae': train_mae,
            'valid_mae': valid_mae,
            'train_graph_mae': train_graph_mae if is_adaptive else float('nan'),
            'valid_graph_mae': valid_graph_mae if is_adaptive else float('nan'),
            'train_mean_alpha': train_alpha if is_adaptive else float('nan'),
            'valid_mean_alpha': valid_alpha if is_adaptive else float('nan'),
            'train_mean_abs_scaled_delta': train_scaled_delta if is_adaptive else float('nan'),
            'valid_mean_abs_scaled_delta': valid_scaled_delta if is_adaptive else float('nan'),
        })
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/valid', valid_mae, epoch)
        log_str = (
            f'epoch:{epoch:0>3}; train:mae:{train_mae:.8f};'
            f'valid:mae:{valid_mae:.8f};'
        )
        if is_adaptive:
            log_str += (
                f' graph_train:mae:{train_graph_mae:.8f};'
                f'graph_valid:mae:{valid_graph_mae:.8f};'
                f'mean_alpha:{valid_alpha:.6f};'
                f'mean_abs_scaled_delta:{valid_scaled_delta:.8f};'
            )
        logger.info(log_str)
        model_saver.step(model, valid_mae, epoch)
        if model_saver.stop and num_epoch - epoch < 50:
            logger.info("Early stopping!")
            break

        try:
            if (epoch + 1) % 50 == 0:
                para_file_name = f'{model_config.model_name}_{data_config.getTargetName()}_epoch{epoch + 1}_para.pth'
                torch.save(model.state_dict(), osp.join(parameter_dir, para_file_name))
                checkpoint = {
                    'model_saver': model_saver.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(checkpoint, osp.join(parameter_dir, 'checkpoint.pt'))

        except Exception:
            logger.error('Checkpoint save failed!')
        # end each epoch

    writer.close()
    end_time = time.time()
    logger.info(f'Train time:{(end_time - start_time):.3f}s')
    logger.info('\n' + model_saver.info())
    # 保存训练曲线到 CSV
    if train_metrics_rows:
        csv_dir = run_dir if run_dir else osp.join('Pic-log', 'train_curves')
        os.makedirs(csv_dir, exist_ok=True)
        csv_name = f'{model_config.model_name}_{data_config.getTargetName()}_train_metrics.csv'
        csv_path = osp.join(csv_dir, csv_name)
        df = pd.DataFrame(train_metrics_rows)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Train metrics (train_loss, valid_loss, train_mae, valid_mae) saved to {csv_path}")
    return model_saver


def test(model, logger, test_loader, mean, std, model_saver: Optional[TopKModelSaver] = None, run_dir: str = None):
    test_mae_list = []
    all_results = []  # [(targets, predictions, emb_crystal, emb_text, graph_predictions, delta_texts, alphas), ...]
    model.eval()
    para_file_name = f'{model_config.model_name}_{data_config.getTargetName()}_para.pth'
    parameter_dir = osp.join(run_dir, 'parameter') if model_saver and run_dir else 'parameter'
    os.makedirs(parameter_dir, exist_ok=True)
    para_file_path = osp.join(parameter_dir, para_file_name)
    logger.info("Testing...")
    collect_embeddings = model_config.model_name in (
        'SFTGNNMultimodal',
        'TextResidualSHFMat',
        'AdaptiveGraphPreservingTextResidualSHFMat',
    )
    with torch.no_grad():
        if model_saver:
            # The test set is evaluated exactly once from the checkpoint with
            # minimum validation MAE. Later validation-ranked candidates are
            # never screened on test data.
            model_paras = model_saver.top_k[:1]
        else:
            logger.info("Test with a pre-trained model")
            model_paras = []
            if osp.exists(para_file_path):
                pretrain_para = torch.load(para_file_path)
                model_paras.append(pretrain_para)
            else:
                raise RuntimeError(
                    f"Tested with a pre-trained model, but no relevant files were found {para_file_path}")

        for model_para in model_paras:
            targets = []
            predictions = []
            graph_predictions = []
            delta_texts = []
            alphas = []
            emb_crystal_list = []
            emb_text_list = []
            state = model_para if isinstance(model_para, dict) else getattr(model_para, 'model_para', model_para)
            model.load_state_dict(state)
            start_time = time.time()
            with tqdm(test_loader, leave=False, desc=f'test') as pbar:
                for batch in pbar:
                    batch = batch.cuda()
                    out = model(batch)
                    pred = out['property_pred'] if isinstance(out, dict) else out
                    pred = pred.squeeze()
                    origin_target = (batch.y.cpu().numpy().flatten() * std + mean).tolist()
                    origin_pred = (pred.cpu().numpy().flatten() * std + mean).tolist()
                    targets.extend(origin_target)
                    predictions.extend(origin_pred)
                    if isinstance(out, dict) and 'pred_graph' in out:
                        graph_pred = out['pred_graph'].squeeze()
                        origin_graph_pred = (graph_pred.cpu().numpy().flatten() * std + mean).tolist()
                        graph_predictions.extend(origin_graph_pred)
                    if isinstance(out, dict) and 'delta_text' in out:
                        delta = out['delta_text'].squeeze().cpu().numpy().flatten().tolist()
                        delta_texts.extend(delta)
                    if isinstance(out, dict) and 'alpha' in out:
                        alpha_values = out['alpha'].detach().cpu().reshape(-1).numpy().tolist()
                        if len(alpha_values) == 1:
                            alpha_values *= len(origin_pred)
                        alphas.extend(alpha_values)
                    if collect_embeddings and isinstance(out, dict) and 'embeddings' in out:
                        emb_crystal_list.append(out['embeddings']['crystal'].cpu().numpy())
                        emb_text_list.append(out['embeddings']['text'].cpu().numpy())
            end_time = time.time()
            test_mae = np.abs(np.array(targets) - np.array(predictions)).mean()
            test_mae_list.append(test_mae)
            emb_crystal = np.vstack(emb_crystal_list) if emb_crystal_list else None
            emb_text = np.vstack(emb_text_list) if emb_text_list else None
            all_results.append((targets, predictions, emb_crystal, emb_text, graph_predictions, delta_texts, alphas))
            log_str = (
                f"Test time:{(end_time - start_time):.3f}s, "
                + (str(model_para) if model_saver else "Pre-trained models")
                + f', Test MAE:{test_mae:.8f};'
            )
            logger.info(log_str)

    logger.info('\n' + data_config.info())
    logger.info('\n' + model_config.info())
    if model_saver:
        logger.info(model_config.message)
        best_mae_index = 0
        best_mae_value = test_mae_list[0]
        logger.info(f"Validation-selected Test MAE:{best_mae_value}\n")
        try:
            selected = model_saver.top_k[0]
            torch.save(selected, para_file_path)
            top1_name = (
                f'{model_config.model_name}_{data_config.getTargetName()}'
                f'_top1_epoch{int(selected.epoch):03d}'
                f'_valid{float(selected.score):.6f}'
                f'_test{float(best_mae_value):.6f}_para.pth'
            )
            torch.save(selected, osp.join(parameter_dir, top1_name))
        except Exception:
            logger.error('The best model save failed!')
        best_targets, best_predictions = all_results[best_mae_index][0], all_results[best_mae_index][1]
        best_emb_crystal = all_results[best_mae_index][2]
        best_emb_text = all_results[best_mae_index][3]
        best_graph_predictions = all_results[best_mae_index][4]
        best_delta_texts = all_results[best_mae_index][5]
        best_alphas = all_results[best_mae_index][6]
    else:
        best_targets, best_predictions = all_results[0][0], all_results[0][1]
        best_emb_crystal = all_results[0][2]
        best_emb_text = all_results[0][3]
        best_graph_predictions = all_results[0][4]
        best_delta_texts = all_results[0][5]
        best_alphas = all_results[0][6]

    # 保存测试集预测和真值到 CSV（含残差，可直接用于残差直方图/箱线图）
    csv_dir = run_dir if run_dir else osp.join('Pic-log', 'test_results')
    if not osp.exists(csv_dir):
        os.makedirs(csv_dir)
    y_true_arr = np.array(best_targets)
    y_pred_arr = np.array(best_predictions)
    residual_arr = y_pred_arr - y_true_arr
    abs_error_arr = np.abs(residual_arr)
    ss_tot = ((y_true_arr - y_true_arr.mean()) ** 2).sum()
    csv_name = f'{model_config.model_name}_{data_config.getTargetName()}_test_pred_true.csv'
    csv_path = osp.join(csv_dir, csv_name)
    df_data = {
        'index': range(len(best_targets)),
        'y_true': best_targets,
        'y_pred': best_predictions,
        'residual': residual_arr,
        'abs_error': abs_error_arr,
    }
    graph_mae = graph_rmse = graph_r2 = float('nan')
    correction_abs_mean = correction_corr = improved_sample_ratio = float('nan')
    if best_graph_predictions:
        y_graph_arr = np.array(best_graph_predictions)
        graph_residual_arr = y_graph_arr - y_true_arr
        graph_abs_error_arr = np.abs(graph_residual_arr)
        graph_mae = graph_abs_error_arr.mean()
        graph_rmse = np.sqrt((graph_residual_arr ** 2).mean())
        graph_r2 = 1 - (graph_residual_arr ** 2).sum() / ss_tot if ss_tot != 0 else float('nan')
        df_data.update({
            'y_graph': y_graph_arr,
            'graph_residual': graph_residual_arr,
            'graph_abs_error': graph_abs_error_arr,
            'text_residual': y_pred_arr - y_graph_arr,
        })
        correction = y_pred_arr - y_graph_arr
        true_graph_residual = y_true_arr - y_graph_arr
        correction_abs_mean = float(np.mean(np.abs(correction)))
        if np.std(correction) > 0 and np.std(true_graph_residual) > 0:
            correction_corr = float(np.corrcoef(correction, true_graph_residual)[0, 1])
        improved_sample_ratio = float(np.mean(abs_error_arr < graph_abs_error_arr))
    if best_delta_texts:
        df_data['delta_text_norm'] = np.array(best_delta_texts)
    if best_alphas:
        df_data['alpha'] = np.array(best_alphas)
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Test predictions and ground truth saved to {csv_path}")

    # 计算并保存测试集汇总指标：MAE、RMSE、R²
    test_mae_final = abs_error_arr.mean()
    test_rmse = np.sqrt((residual_arr ** 2).mean())
    ss_res = (residual_arr ** 2).sum()
    test_r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    logger.info(f"Test MAE: {test_mae_final:.8f}, RMSE: {test_rmse:.8f}, R²: {test_r2:.8f}")
    if best_graph_predictions:
        logger.info(f"Graph-only Test MAE: {graph_mae:.8f}, RMSE: {graph_rmse:.8f}, R²: {graph_r2:.8f}")
        logger.info(
            f"Residual diagnostics: mean|correction|={correction_abs_mean:.8f}, "
            f"corr(correction,target-pred_graph)={correction_corr:.6f}, "
            f"improved_sample_ratio={improved_sample_ratio:.6f}, "
            f"mean_alpha={float(np.mean(best_alphas)) if best_alphas else float('nan'):.6f}"
        )

    metrics_name = f'{model_config.model_name}_{data_config.getTargetName()}_test_metrics.csv'
    metrics_dir = run_dir if run_dir else csv_dir
    metrics_path = osp.join(metrics_dir, metrics_name)
    df_metrics = pd.DataFrame([{
        'model': model_config.model_name,
        'target': data_config.getTargetName(),
        'MAE': test_mae_final,
        'RMSE': test_rmse,
        'R2': test_r2,
        'graph_MAE': graph_mae,
        'graph_RMSE': graph_rmse,
        'graph_R2': graph_r2,
        'alpha': float(np.mean(best_alphas)) if best_alphas else float('nan'),
        'mean_abs_correction': correction_abs_mean,
        'correction_target_corr': correction_corr,
        'improved_sample_ratio': improved_sample_ratio,
        'num_samples': len(best_targets),
    }])
    df_metrics.to_csv(metrics_path, index=False, encoding='utf-8')
    logger.info(f"Test metrics (MAE, RMSE, R²) saved to {metrics_path}")

    # 多模态时保存 crystal/text/multimodal embedding 供 t-SNE 使用
    if collect_embeddings and best_emb_crystal is not None and best_emb_text is not None:
        emb_dir = osp.join(run_dir, 'embeddings') if run_dir else osp.join('Pic-log', 'test_results', 'embeddings')
        if not osp.exists(emb_dir):
            os.makedirs(emb_dir)
        emb_prefix = f'{model_config.model_name}_{data_config.getTargetName()}_test'
        emb_multimodal = np.concatenate([best_emb_crystal, best_emb_text], axis=1)
        np.savez(
            osp.join(emb_dir, f'{emb_prefix}_embeddings.npz'),
            crystal_embedding=best_emb_crystal,
            text_embedding=best_emb_text,
            multimodal_embedding=emb_multimodal,
        )
        logger.info(f"Embeddings (crystal, text, multimodal) saved to {emb_dir}/{emb_prefix}_embeddings.npz for t-SNE")


def _save_validation_best(model_saver: TopKModelSaver, run_dir: str, logger) -> None:
    selected = model_saver.top_k[0]
    parameter_dir = osp.join(run_dir, 'parameter')
    os.makedirs(parameter_dir, exist_ok=True)
    filename = (
        f'{model_config.model_name}_{data_config.getTargetName()}'
        f'_validation_best_epoch{int(selected.epoch):03d}'
        f'_valid{float(selected.score):.8f}.pth'
    )
    torch.save(selected, osp.join(parameter_dir, filename))
    logger.info(
        f'Validation-only best MAE:{float(selected.score):.8f}; '
        f'epoch:{int(selected.epoch)}; checkpoint:{osp.join(parameter_dir, filename)}'
    )


def train_and_test():
    random.seed(model_config.random_seed)
    np.random.seed(model_config.random_seed)
    torch.manual_seed(model_config.random_seed)
    torch.cuda.manual_seed(model_config.random_seed)
    torch.cuda.manual_seed_all(model_config.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    is_multimodal = model_config.model_name in (
        'SFTGNNMultimodal',
        'TextResidualSHFMat',
        'AdaptiveGraphPreservingTextResidualSHFMat',
    )
    if is_multimodal:
        data_config.use_multimodal = True
        t = getattr(model_config, 'text_emb_path', '')
        if t:
            data_config.text_emb_path = t

    match model_config.model_name:
        case "SFTGNN":
            model = SFTGNN(num_layers=model_config.num_layers, in_dim=data_config.getAtomDim())
        case "SFTGNNMultimodal":
            backbone = SFTGNN(num_layers=model_config.num_layers, in_dim=data_config.getAtomDim())
            model = SFTGNNMultimodal(
                sftgnn=backbone,
                crystal_dim=getattr(model_config, 'node_feature', 128),
                latent_dim=getattr(model_config, 'latent_dim', 128),
                text_in_dim=getattr(model_config, 'text_emb_dim', 768),
                text_hidden_dim=getattr(model_config, 'text_hidden_dim', 256),
                text_out_dim=getattr(model_config, 'text_out_dim', 64),
                property_fusion=getattr(model_config, 'property_fusion', 'crystal_only'),
            )
        case "TextResidualSHFMat":
            backbone = SFTGNN(num_layers=model_config.num_layers, in_dim=data_config.getAtomDim())
            model = TextResidualSHFMat(
                sftgnn=backbone,
                graph_dim=getattr(model_config, 'node_feature', 128),
                text_in_dim=getattr(model_config, 'text_emb_dim', 768),
                text_hidden_dim=getattr(model_config, 'text_hidden_dim', 128),
                residual_hidden_dim=getattr(model_config, 'text_residual_hidden_dim', 128),
                alpha_init=getattr(model_config, 'text_residual_alpha_init', 0.03),
                alpha_max=getattr(model_config, 'text_residual_alpha_max', 0.2),
                dropout=getattr(model_config, 'text_residual_dropout', 0.1),
            )
        case "AdaptiveGraphPreservingTextResidualSHFMat":
            backbone = SFTGNN(num_layers=model_config.num_layers, in_dim=data_config.getAtomDim())
            model = AdaptiveGraphPreservingTextResidualSHFMat(
                sftgnn=backbone,
                graph_dim=getattr(model_config, 'node_feature', 128),
                text_in_dim=getattr(model_config, 'text_emb_dim', 768),
                text_hidden_dim=getattr(model_config, 'text_hidden_dim', 128),
                residual_hidden_dim=getattr(model_config, 'text_residual_hidden_dim', 128),
                gate_hidden_dim=getattr(model_config, 'adaptive_gate_hidden_dim', 128),
                alpha_init=getattr(model_config, 'adaptive_alpha_init', 0.02),
                alpha_max=getattr(model_config, 'adaptive_alpha_max', 0.1),
                alpha_floor=getattr(model_config, 'adaptive_alpha_floor', 0.0),
                correction_mode=getattr(model_config, 'adaptive_correction_mode', 'learned'),
                dropout=getattr(model_config, 'text_residual_dropout', 0.1),
                text_dropout=getattr(model_config, 'adaptive_text_dropout', 0.2),
            )
        case _:
            raise ValueError(f'Invalid model name: {model_config.model_name}')

    model.cuda()
    run_dir = create_run_dir()
    logger = getLogger(run_dir)
    logger.info(f"Run output directory: {run_dir}")
    train_loader, valid_loader, test_loader, mean, std = get_dataloader()
    model_saver = TopKModelSaver(model.state_dict(), k=5)

    train(model, logger, train_loader, valid_loader, mean, std, model_saver, run_dir)
    if getattr(model_config, 'validation_only', False):
        _save_validation_best(model_saver, run_dir, logger)
    else:
        test(model, logger, test_loader, mean, std, model_saver, run_dir)


def test_with_pretrained_model():
    is_multimodal = model_config.model_name in (
        'SFTGNNMultimodal',
        'TextResidualSHFMat',
        'AdaptiveGraphPreservingTextResidualSHFMat',
    )
    if is_multimodal:
        data_config.use_multimodal = True
        t = getattr(model_config, 'text_emb_path', '')
        if t:
            data_config.text_emb_path = t

    match model_config.model_name:
        case "SFTGNN":
            model = SFTGNN(num_layers=model_config.num_layers, in_dim=data_config.getAtomDim())
        case "SFTGNNMultimodal":
            backbone = SFTGNN(num_layers=model_config.num_layers, in_dim=data_config.getAtomDim())
            model = SFTGNNMultimodal(
                sftgnn=backbone,
                crystal_dim=getattr(model_config, 'node_feature', 128),
                latent_dim=getattr(model_config, 'latent_dim', 128),
                text_in_dim=getattr(model_config, 'text_emb_dim', 768),
                text_hidden_dim=getattr(model_config, 'text_hidden_dim', 256),
                text_out_dim=getattr(model_config, 'text_out_dim', 64),
                property_fusion=getattr(model_config, 'property_fusion', 'crystal_only'),
            )
        case "TextResidualSHFMat":
            backbone = SFTGNN(num_layers=model_config.num_layers, in_dim=data_config.getAtomDim())
            model = TextResidualSHFMat(
                sftgnn=backbone,
                graph_dim=getattr(model_config, 'node_feature', 128),
                text_in_dim=getattr(model_config, 'text_emb_dim', 768),
                text_hidden_dim=getattr(model_config, 'text_hidden_dim', 128),
                residual_hidden_dim=getattr(model_config, 'text_residual_hidden_dim', 128),
                alpha_init=getattr(model_config, 'text_residual_alpha_init', 0.03),
                alpha_max=getattr(model_config, 'text_residual_alpha_max', 0.2),
                dropout=getattr(model_config, 'text_residual_dropout', 0.1),
            )
        case "AdaptiveGraphPreservingTextResidualSHFMat":
            backbone = SFTGNN(num_layers=model_config.num_layers, in_dim=data_config.getAtomDim())
            model = AdaptiveGraphPreservingTextResidualSHFMat(
                sftgnn=backbone,
                graph_dim=getattr(model_config, 'node_feature', 128),
                text_in_dim=getattr(model_config, 'text_emb_dim', 768),
                text_hidden_dim=getattr(model_config, 'text_hidden_dim', 128),
                residual_hidden_dim=getattr(model_config, 'text_residual_hidden_dim', 128),
                gate_hidden_dim=getattr(model_config, 'adaptive_gate_hidden_dim', 128),
                alpha_init=getattr(model_config, 'adaptive_alpha_init', 0.02),
                alpha_max=getattr(model_config, 'adaptive_alpha_max', 0.1),
                alpha_floor=getattr(model_config, 'adaptive_alpha_floor', 0.0),
                correction_mode=getattr(model_config, 'adaptive_correction_mode', 'learned'),
                dropout=getattr(model_config, 'text_residual_dropout', 0.1),
                text_dropout=getattr(model_config, 'adaptive_text_dropout', 0.2),
            )
        case _:
            raise ValueError(f'Invalid model name: {model_config.model_name}')

    model.cuda()
    run_dir = create_run_dir()
    logger = getLogger(run_dir)
    logger.info(f"Run output directory: {run_dir}")
    train_loader, valid_loader, test_loader, mean, std = get_dataloader()
    test(model, logger, test_loader, mean, std, model_saver=None, run_dir=run_dir)


if __name__ == '__main__':
    # If you don't want to set up training configurations using command-line parameters, you can set up custom configurations here. However, this will work for all training.
    # If you want to use different training configurations for each task, you can use TrainConfigManager
    # data_config.batch_size = 32
    # data_config.atom_features = AtomFeatureType.CGCNN
    # model_config.message = "SFTGNN Standard Model"
    # model_config.lr = 1e-3
    # model_config.max_lr = 1e-3
    # model_config.begin_epoch = 0
    # model_config.num_epoch = 500

    # You can add the tasks you want to train within targets
    # targets = [*JarvisTarget, *MPTarget]  #All tasks for training both datasets
    # targets = [*JarvisTarget] #All tasks to train the Jarvis dataset
    # targets = [*MPTarget] #All tasks to train the MP dataset
    targets = [MPTarget.BulkModuli]
    for t in targets:
        data_config.target = t
        with TrainConfigManager(t):
            train_and_test()
            # If you just want to test with a pre-trained model, you can replace it with a function like this
            # test_with_pretrained_model()
