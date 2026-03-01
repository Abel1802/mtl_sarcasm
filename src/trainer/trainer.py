import os
import logging
import torch
import torch.nn as nn
import wandb
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr

class MTLTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        config,
        device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # 定义多任务损失函数
        # 注意：分类通常建议用 BCEWithLogitsLoss，模型最后一层不需要 Sigmoid，数值更稳定
        self.criterion_cls = nn.BCEWithLogitsLoss() 
        self.criterion_reg = nn.MSELoss()
        self.lambda_reg = config.get('lambda_reg', 0.1) # 控制辅助任务的权重
        
        # 训练超参数
        self.epochs = config.get('epochs', 50)
        self.clip_grad = config.get('clip_grad', 1.0)

        # 保存模型路径
        self.save_dir = config.get('save_dir', 'output')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置 Scheduler with Warmup
        total_steps = len(self.train_loader) * self.epochs
        warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Early Stopping 机制
        self.patience = config.get('patience', 5)
        self.best_val_loss = float('inf')
        self.early_step = 0
        
        # 初始化 WandB
        if config.get('use_wandb', True):
            wandb.init(
                project=config.get('wandb_project', 'MTL_Sarcasm_Detection'),
                config=config,
                name=config.get('run_name', 'baseline_run')
            )
            wandb.watch(self.model, log="all", log_freq=100)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_cls_loss, total_reg_loss = 0, 0, 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        
        for batch in progress_bar:
            # 1. 提取数据并送入设备
            features = {
                'text': batch['text'].to(self.device),
                'audio': batch['audio'].to(self.device),
                'video': batch['video'].to(self.device)
            }
            # --- 增加的模态消融 (Ablation) 执行逻辑 ---
            if self.config.get('ablate_text', False):
                features['text'] = torch.zeros_like(features['text'])
            if self.config.get('ablate_audio', False):
                features['audio'] = torch.zeros_like(features['audio'])
            if self.config.get('ablate_video', False):
                features['video'] = torch.zeros_like(features['video'])
            # ------------------------------------------
            labels = batch['label'].to(self.device).float()       # [batch_size]
            certainties = batch['certainty'].to(self.device).float() # [batch_size]
            
            # 2. 前向传播
            self.optimizer.zero_grad()
            cls_logits, reg_preds = self.model(features)
            
            # 3. 计算多任务 Loss
            loss_cls = self.criterion_cls(cls_logits.squeeze(), labels)
            loss_reg = self.criterion_reg(reg_preds.squeeze(), certainties)
            loss = loss_cls + self.lambda_reg * loss_reg
            
            # 4. 反向传播与优化
            loss.backward()
            
            # 梯度裁剪 (防止多模态 Transformer 出现梯度爆炸)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 5. 记录日志
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if wandb.run is not None:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_cls_loss": loss_cls.item(),
                    "train/batch_reg_loss": loss_reg.item(),
                    "train/lr": self.scheduler.get_last_lr()[0]
                })
            # logging.info(f"Epoch {epoch}/{self.epochs} [Train] Loss: {total_loss:.4f}, Cls Loss: {total_cls_loss:.4f}, Reg Loss: {total_reg_loss:.4f}")

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.model.eval()
        total_loss, total_cls_loss, total_reg_loss = 0, 0, 0
        
        all_labels, all_cls_preds = [], []
        all_certainties, all_reg_preds = [], []
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")
        
        for batch in progress_bar:
            features = {
                'text': batch['text'].to(self.device),
                'audio': batch['audio'].to(self.device),
                'video': batch['video'].to(self.device)
            }
            # --- 增加的模态消融 (Ablation) 执行逻辑 ---
            if self.config.get('ablate_text', False):
                features['text'] = torch.zeros_like(features['text'])
            if self.config.get('ablate_audio', False):
                features['audio'] = torch.zeros_like(features['audio'])
            if self.config.get('ablate_video', False):
                features['video'] = torch.zeros_like(features['video'])
            # ------------------------------------------
            labels = batch['label'].to(self.device).float()
            certainties = batch['certainty'].to(self.device).float()
            
            cls_logits, reg_preds = self.model(features)
            
            loss_cls = self.criterion_cls(cls_logits.squeeze(), labels)
            loss_reg = self.criterion_reg(reg_preds.squeeze(), certainties)
            loss = loss_cls + self.lambda_reg * loss_reg
            
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            
            # 收集预测结果用于计算 Metrics
            cls_probs = torch.sigmoid(cls_logits).squeeze()
            cls_preds = (cls_probs > 0.5).int()
            
            all_labels.extend(labels.cpu().numpy())
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_certainties.extend(certainties.cpu().numpy())
            all_reg_preds.extend(reg_preds.squeeze().cpu().numpy())

        # 计算评估指标
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_cls_preds)
        f1 = f1_score(all_labels, all_cls_preds, average='macro')
        mae = mean_absolute_error(all_certainties, all_reg_preds)
        pearson_corr, _ = pearsonr(all_certainties, all_reg_preds)
        
        metrics = {
            "val/loss": avg_loss,
            "val/cls_loss": total_cls_loss / len(self.val_loader),
            "val/reg_loss": total_reg_loss / len(self.val_loader),
            "val/acc": acc,
            "val/macro_f1": f1,
            "val/mae": mae,
            "val/pearson": pearson_corr
        }
        
        if wandb.run is not None:
            wandb.log(metrics)
            
        logging.info(f"\nVal Epoch {epoch}: Loss={avg_loss:.4f}, ACC={acc:.4f}, F1={f1:.4f}, MAE={mae:.4f}, Pearson={pearson_corr:.4f}\n")
        return avg_loss, f1

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_f1 = self.eval_epoch(epoch)
            
            # Early Stopping & Checkpointing 逻辑
            # 这里以 val_loss 为标准，也可以改为 val_f1 (如果是越大约好，逻辑要反过来)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_step = 0
                
                # 保存最佳模型
                save_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                logging.info(f"🔥 Best model saved at epoch {epoch} with Val Loss: {val_loss:.4f}")
            else:
                self.early_step += 1
                logging.info(f"⚠️ Early stopping counter: {self.early_step} out of {self.patience}")
                
            if self.early_step >= self.patience:
                logging.info("🛑 Early stopping triggered. Training stopped.")
                break
                
        if wandb.run is not None:
            wandb.finish()