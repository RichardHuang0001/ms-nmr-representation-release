# trainer.py
# <-- [核心] 封装训练、验证循环和指标计算
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import logging
from pathlib import Path
import sys
import time

class Trainer:
    """
    训练器类，封装了所有与模型训练和评估相关的逻辑。
    这使得主训练脚本可以保持简洁，只负责对象的初始化和启动训练流程。
    """
    def __init__(self, model, optimizer, train_loader, val_loader, config):
        """
        初始化训练器。
        
        :param model: 要训练的PyTorch模型。
        :param optimizer: 用于更新模型参数的优化器。
        :param train_loader: 训练数据的DataLoader。
        :param val_loader: 验证数据的DataLoader。
        :param config: 包含所有超参数和设置的配置对象 (Box)。
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        # 自动检测并选择可用的设备（优先使用GPU）
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info(f"✅ 模型和数据将使用设备: {self.device}")

        # 初始化 Weights & Biases (W&B) 用于实验跟踪
        # `project_name` 和 `run_name` 用于在W&B仪表板上组织实验
        wandb.init(project=config.wandb.project_name, name=config.wandb.run_name, config=config)
        # `wandb.watch` 会自动记录模型的梯度和参数，便于调试和分析
        wandb.watch(self.model, log_freq=int(getattr(self.config.wandb, "watch_log_freq", 1000)))

    def _calculate_loss(self, predictions, labels, masked_inputs=None):
        """
        计算用于预训练任务的混合损失（改进版）。
        返回：
          - total_loss: 用于反向传播的总损失
          - loss_parts: dict，记录各子损失（用于日志）
          - stats: dict，详细的per-modality和per-strategy统计
        """
        # 1. 找到有任意被掩码的峰（行），作为总体候选
        any_mask = (labels != -100).any(dim=-1)  # [B, L] 布尔

        # 如果整个 batch 没有任何被掩码位置，返回 0
        if not any_mask.any():
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            empty_stats = {
                "modality_count": {"h_nmr": 0, "c_nmr": 0, "ms": 0, "unknown": 0},
                "strategy_count": {"position": 0, "y_axis": 0, "coupling": 0, "modality": 0, "unknown": 0},
                "modality_loss": {"h_nmr": 0.0, "c_nmr": 0.0, "ms": 0.0},
            }
            return zero, {"numeric": 0.0, "modality": 0.0, "multiplicity": 0.0}, empty_stats

        # 把候选位置展开为行索引
        active_preds_all = predictions[any_mask]  # [N, D]
        active_labels_all = labels[any_mask]      # [N, D]
        
        # 统计信息：模态和mask策略
        stats = {
            "modality_count": {"h_nmr": 0, "c_nmr": 0, "ms": 0, "unknown": 0},
            "strategy_count": {"position": 0, "y_axis": 0, "coupling": 0, "modality": 0, "unknown": 0},
            "modality_loss": {"h_nmr": 0.0, "c_nmr": 0.0, "ms": 0.0}
        }
        
        # 如果有masked_inputs，分析每个被mask的峰的模态和策略（向量化实现，避免逐峰Python循环导致GPU同步变慢）
        masked_inputs_flat = None
        if masked_inputs is not None:
            masked_inputs_flat = masked_inputs[any_mask]  # [N, D]

            # ----- 模态统计 -----
            modality_idx = torch.argmax(masked_inputs_flat[:, :8], dim=-1)  # [N]
            is_h = modality_idx == 0
            is_c = modality_idx == 1
            is_ms = (modality_idx >= 2) & (modality_idx <= 7)
            is_unknown_mod = ~(is_h | is_c | is_ms)

            stats["modality_count"]["h_nmr"] = int(is_h.sum().detach().item())
            stats["modality_count"]["c_nmr"] = int(is_c.sum().detach().item())
            stats["modality_count"]["ms"] = int(is_ms.sum().detach().item())
            stats["modality_count"]["unknown"] = int(is_unknown_mod.sum().detach().item())

            # ----- mask策略统计（严格复刻原判断逻辑的向量化版本）-----
            row_mask = active_labels_all != -100  # [N, D]

            # 1) modality: masked_dims 只在 [0,7]
            strat_modality = row_mask[:, 0:8].any(dim=-1) & (~row_mask[:, 8:].any(dim=-1))

            # 2) position: masked_dims == [8]
            strat_position = row_mask[:, 8] & (~row_mask[:, 0:8].any(dim=-1)) & (~row_mask[:, 9:].any(dim=-1))

            # 3) y_axis: masked_dims == {9,10,11}
            strat_y_axis = row_mask[:, 9:12].all(dim=-1) & (~row_mask[:, 0:9].any(dim=-1)) & (~row_mask[:, 12:].any(dim=-1))

            # 4) coupling: min(masked_dims) >= 12
            strat_coupling = row_mask[:, 12:].any(dim=-1) & (~row_mask[:, 0:12].any(dim=-1))

            strat_known = strat_modality | strat_position | strat_y_axis | strat_coupling
            strat_unknown = ~strat_known

            stats["strategy_count"]["modality"] = int(strat_modality.sum().detach().item())
            stats["strategy_count"]["position"] = int(strat_position.sum().detach().item())
            stats["strategy_count"]["y_axis"] = int(strat_y_axis.sum().detach().item())
            stats["strategy_count"]["coupling"] = int(strat_coupling.sum().detach().item())
            stats["strategy_count"]["unknown"] = int(strat_unknown.sum().detach().item())

        # 3. 从配置文件中获取特征向量不同部分的切片索引
        try:
            feature_slices = self.config.peak_vector.feature_slices
        except Exception:
            raise KeyError("配置文件中缺少 peak_vector.feature_slices，请检查 config.yaml")

        # 模态分类部分
        s_mod, e_mod = feature_slices.modality

        # 连续值部分（位置+强度+积分+宽度+J值）
        s_num = feature_slices.position[0]
        e_num = feature_slices.j_count[1]

        # 多重性分类部分
        s_mult, e_mult = feature_slices.multiplicity

        # ----- 数值 z-score 裁剪阈值（可选） -----
        clip_z = None
        training_cfg = getattr(self.config, "training", None)
        if training_cfg is not None:
            try:
                clip_val = float(getattr(training_cfg, "numeric_z_clip", 0.0))
                if clip_val > 0:
                    clip_z = clip_val
            except Exception:
                clip_z = None
        # -------------------------------------

        total_loss = torch.tensor(0.0, device=self.device)

        # 记录各子损失（用于 wandb）
        numeric_loss_val = 0.0
        modality_loss_val = 0.0
        multiplicity_loss_val = 0.0

        # ---------- 数值部分 (MSE) ----------
        num_mask = (active_labels_all[:, s_num:e_num] != -100).any(dim=-1)
        preds_num = None
        labels_num = None
        if num_mask.any():
            preds_num = active_preds_all[num_mask, s_num:e_num]
            labels_num = active_labels_all[num_mask, s_num:e_num]

            # 数值标签裁剪，避免极端 outlier
            if clip_z is not None:
                labels_num = torch.clamp(labels_num, -clip_z, clip_z)

            numeric_loss = F.mse_loss(preds_num, labels_num)
            total_loss = total_loss + numeric_loss
            numeric_loss_val = float(numeric_loss.detach().item())

        # ---------- 模态部分 (CrossEntropy) ----------
        mod_mask = (active_labels_all[:, s_mod:e_mod] != -100).any(dim=-1)
        if mod_mask.any():
            preds_mod = active_preds_all[mod_mask, s_mod:e_mod]          # logits [M, C]
            labels_mod_onehot = active_labels_all[mod_mask, s_mod:e_mod] # one-hot或one-hot-like
            target_mod = torch.argmax(labels_mod_onehot, dim=-1).long().to(self.device)
            modality_loss = F.cross_entropy(preds_mod, target_mod)
            total_loss = total_loss + modality_loss
            modality_loss_val = float(modality_loss.detach().item())

        # ---------- 多重性部分 (CrossEntropy) ----------
        mult_mask = (active_labels_all[:, s_mult:e_mult] != -100).any(dim=-1)
        if mult_mask.any():
            preds_mult = active_preds_all[mult_mask, s_mult:e_mult]
            labels_mult_onehot = active_labels_all[mult_mask, s_mult:e_mult]
            target_mult = torch.argmax(labels_mult_onehot, dim=-1).long().to(self.device)
            multiplicity_loss = F.cross_entropy(preds_mult, target_mult)
            total_loss = total_loss + multiplicity_loss
            multiplicity_loss_val = float(multiplicity_loss.detach().item())

        # 计算per-modality的numeric loss（用于统计）
        if masked_inputs_flat is not None and num_mask.any() and preds_num is not None and labels_num is not None:
            # 每个峰的 numeric MSE（按特征维度取均值），再按模态做求和统计
            mse_per_dim = F.mse_loss(preds_num, labels_num, reduction="none")  # [K, d_num]
            mse_per_peak = mse_per_dim.mean(dim=-1)  # [K]
            modality_idx_num = torch.argmax(masked_inputs_flat[num_mask, :8], dim=-1)  # [K]

            sum_h = mse_per_peak[modality_idx_num == 0].sum()
            sum_c = mse_per_peak[modality_idx_num == 1].sum()
            sum_ms = mse_per_peak[(modality_idx_num >= 2) & (modality_idx_num <= 7)].sum()

            stats["modality_loss"]["h_nmr"] = float(sum_h.detach().item())
            stats["modality_loss"]["c_nmr"] = float(sum_c.detach().item())
            stats["modality_loss"]["ms"] = float(sum_ms.detach().item())

        loss_parts = {
            "numeric": numeric_loss_val,
            "modality": modality_loss_val,
            "multiplicity": multiplicity_loss_val,
        }

        return total_loss, loss_parts, stats
        
    def _run_epoch(self, dataloader, is_training=True):
        """
        运行一个完整的epoch，可以是训练或验证。

        :param dataloader: 用于该epoch的数据加载器。
        :param is_training: 布尔值，如果为True，则执行训练步骤（反向传播和优化）。
        :return:
            - avg_loss: 该 epoch 的平均总损失
            - avg_parts: dict，记录各子损失的 epoch 平均值
        """
        self.model.train(is_training)
        total_loss = 0.0

        # 累计各子损失
        sum_numeric = 0.0
        sum_modality = 0.0
        sum_multiplicity = 0.0
        
        # 累计详细统计
        epoch_stats = {
            "modality_count": {"h_nmr": 0, "c_nmr": 0, "ms": 0, "unknown": 0},
            "strategy_count": {"position": 0, "y_axis": 0, "coupling": 0, "modality": 0, "unknown": 0},
            "modality_loss": {"h_nmr": 0.0, "c_nmr": 0.0, "ms": 0.0}
        }

        # 使用tqdm创建进度条，方便监控
        desc = "Train" if is_training else "Eval"
        enable_tqdm = getattr(self.config.training, "enable_tqdm", True)
        update_every = int(getattr(self.config.training, "progress_update_every", 100))
        if enable_tqdm:
            progress_bar = tqdm(
                dataloader,
                desc=desc,
                leave=False,
                disable=not sys.stdout.isatty(),
                mininterval=1.0,
            )
            iterator = enumerate(progress_bar)
        else:
            iterator = enumerate(dataloader)

        num_batches = 0

        for i, batch in iterator:
            masked_inputs, masks, labels = [b.to(self.device) for b in batch]

            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                predictions = self.model(masked_inputs, masks)
                loss, loss_parts, batch_stats = self._calculate_loss(predictions, labels, masked_inputs)

            if is_training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            sum_numeric += loss_parts["numeric"]
            sum_modality += loss_parts["modality"]
            sum_multiplicity += loss_parts["multiplicity"]
            
            # 累积统计信息
            for k in epoch_stats["modality_count"]:
                epoch_stats["modality_count"][k] += batch_stats["modality_count"][k]
            for k in epoch_stats["strategy_count"]:
                epoch_stats["strategy_count"][k] += batch_stats["strategy_count"][k]
            for k in epoch_stats["modality_loss"]:
                epoch_stats["modality_loss"][k] += batch_stats["modality_loss"][k]
            
            num_batches += 1

            if enable_tqdm and ((i + 1) % update_every == 0):
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(num_batches, 1)
        avg_parts = {
            "numeric": sum_numeric / max(num_batches, 1),
            "modality": sum_modality / max(num_batches, 1),
            "multiplicity": sum_multiplicity / max(num_batches, 1),
        }
        
        # 计算per-modality的平均loss
        total_peaks = sum(epoch_stats["modality_count"].values())
        if total_peaks > 0:
            for modality in epoch_stats["modality_loss"]:
                count = epoch_stats["modality_count"][modality]
                if count > 0:
                    epoch_stats["modality_loss"][modality] /= count
        
        avg_parts["stats"] = epoch_stats
        return avg_loss, avg_parts
    def _save_checkpoint(self, epoch, val_loss, is_best):
        """
        保存模型检查点。只在验证损失改善时保存“最佳模型”。

        :param epoch: 当前的epoch号。
        :param val_loss: 当前的验证损失。
        :param is_best: 布尔值，指示当前模型是否是迄今为止最好的。
        """
        # 如果当前模型不是最好的，则不执行任何操作
        if not is_best:
            return

        # 确保检查点目录存在
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "best_model.pt"
        
        # 保存模型状态字典、优化器状态、epoch号和验证损失
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        logging.info(f"🚀 新的最佳模型已保存 (Epoch {epoch+1}, Val Loss: {val_loss:.4f}) 至: {checkpoint_path}")

    def train(self):
        """
        执行完整的模型训练流程，包括多个epoch的训练和验证。
        """
        # 初始化最佳验证损失为一个极大值
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.epochs):
            t0_train = time.time()
            train_loss, train_parts = self._run_epoch(self.train_loader, is_training=True)
            t1_train = time.time()
            t0_val = time.time()
            val_loss, val_parts = self._run_epoch(self.val_loader, is_training=False)
            t1_val = time.time()
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

            logging.info(
                f"Epoch {epoch+1}/{self.config.training.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"LR: {lr:.6f} | Train Time: {t1_train - t0_train:.2f}s | "
                f"Val Time: {t1_val - t0_val:.2f}s"
            )
            
            # 打印详细的模态和策略统计（每5个epoch打印一次，避免刷屏）
            if (epoch + 1) % 5 == 0 or epoch == 0:
                train_stats = train_parts.get("stats", {})
                val_stats = val_parts.get("stats", {})
                
                logging.info("  [Train] Modality Distribution:")
                total_train = sum(train_stats.get("modality_count", {}).values())
                for mod, cnt in train_stats.get("modality_count", {}).items():
                    pct = 100.0 * cnt / total_train if total_train > 0 else 0
                    loss_val = train_stats.get("modality_loss", {}).get(mod, 0.0)
                    logging.info(f"    {mod:8s}: {cnt:6d} peaks ({pct:5.1f}%) | Avg Loss: {loss_val:.4f}")
                
                logging.info("  [Train] Mask Strategy Distribution:")
                for strategy, cnt in train_stats.get("strategy_count", {}).items():
                    pct = 100.0 * cnt / total_train if total_train > 0 else 0
                    logging.info(f"    {strategy:10s}: {cnt:6d} ({pct:5.1f}%)")
                
                logging.info("  [Val] Modality Distribution:")
                total_val = sum(val_stats.get("modality_count", {}).values())
                for mod, cnt in val_stats.get("modality_count", {}).items():
                    pct = 100.0 * cnt / total_val if total_val > 0 else 0
                    loss_val = val_stats.get("modality_loss", {}).get(mod, 0.0)
                    logging.info(f"    {mod:8s}: {cnt:6d} peaks ({pct:5.1f}%) | Avg Loss: {loss_val:.4f}")

            # 在 wandb 里记录总 loss 以及分模态 loss
            wandb_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "train/loss_numeric": train_parts["numeric"],
                "train/loss_modality": train_parts["modality"],
                "train/loss_multiplicity": train_parts["multiplicity"],
                "val/loss_numeric": val_parts["numeric"],
                "val/loss_modality": val_parts["modality"],
                "val/loss_multiplicity": val_parts["multiplicity"],
            }
            
            # 添加per-modality的详细统计到wandb
            train_stats = train_parts.get("stats", {})
            val_stats = val_parts.get("stats", {})
            
            for mod in ["h_nmr", "c_nmr", "ms"]:
                wandb_log[f"train/modality_loss_{mod}"] = train_stats.get("modality_loss", {}).get(mod, 0.0)
                wandb_log[f"val/modality_loss_{mod}"] = val_stats.get("modality_loss", {}).get(mod, 0.0)
                wandb_log[f"train/modality_count_{mod}"] = train_stats.get("modality_count", {}).get(mod, 0)
                wandb_log[f"val/modality_count_{mod}"] = val_stats.get("modality_count", {}).get(mod, 0)
            
            for strategy in ["position", "y_axis", "coupling", "modality"]:
                wandb_log[f"train/strategy_count_{strategy}"] = train_stats.get("strategy_count", {}).get(strategy, 0)
                wandb_log[f"val/strategy_count_{strategy}"] = val_stats.get("strategy_count", {}).get(strategy, 0)
            
            wandb.log(wandb_log)

            # 检查当前验证损失是否是历史最佳
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # 根据is_best标志决定是否保存模型
            self._save_checkpoint(epoch, val_loss, is_best)
