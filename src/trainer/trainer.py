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

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import logging
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr
from transformers import get_linear_schedule_with_warmup # 假设你从这里导入的

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
        self.criterion_cls = nn.BCEWithLogitsLoss() 
        self.criterion_reg = nn.MSELoss()
        # ⭐️ 新增：为 Rationale 预测添加多标签分类损失
        self.criterion_rat = nn.BCEWithLogitsLoss() 
        
        # 定义多任务权重
        self.lambda_reg = config.get('lambda_reg', 0.1)
        self.lambda_rat = config.get('lambda_rat', 0.1) # ⭐️ 新增：rationale loss 权重
        
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
        self.best_val_f1 = 0
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
        # ⭐️ 新增：追踪 Rationale Loss
        total_loss, total_cls_loss, total_reg_loss, total_rat_loss = 0, 0, 0, 0
        
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
            # ⭐️ 新增：获取 Rationale 标签 [batch_size, 3]
            rationales = batch['rationale'].to(self.device).float() 
            
            # 2. 前向传播
            self.optimizer.zero_grad()
            # ⭐️ 新增：接收 Rationale 预测输出
            cls_logits, reg_preds, rat_logits = self.model(features)
            
            # 3. 计算多任务 Loss
            loss_cls = self.criterion_cls(cls_logits.squeeze(), labels)
            loss_reg = self.criterion_reg(reg_preds.squeeze(), certainties)
            # ⭐️ 新增：计算 Rationale Loss（注意 rat_logits 和 rationales 都不需要 squeeze，因为都是 [B, 3]）
            loss_rat = self.criterion_rat(rat_logits, rationales)
            
            # ⭐️ 新增：将 Rationale Loss 加入总 Loss
            loss = loss_cls + (self.lambda_reg * loss_reg) + (self.lambda_rat * loss_rat)
            
            # 4. 反向传播与优化
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.scheduler.step()
            
            # 5. 记录日志
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            total_rat_loss += loss_rat.item() # ⭐️ 新增
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rat_l': f"{loss_rat.item():.4f}", # ⭐️ 新增：在进度条显示 Rationale Loss
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if wandb.run is not None:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_cls_loss": loss_cls.item(),
                    "train/batch_reg_loss": loss_reg.item(),
                    "train/batch_rat_loss": loss_rat.item(), # ⭐️ 新增
                    "train/lr": self.scheduler.get_last_lr()[0]
                })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.model.eval()
        total_loss, total_cls_loss, total_reg_loss, total_rat_loss = 0, 0, 0, 0 # ⭐️ 新增
        
        all_labels, all_cls_preds = [], []
        all_certainties, all_reg_preds = [], []
        all_rationales, all_rat_preds = [], [] # ⭐️ 新增：用于存储 rationale 预测结果以计算评估指标
        
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
            rationales = batch['rationale'].to(self.device).float() # ⭐️ 新增
            
            # ⭐️ 新增：接收三个输出
            cls_logits, reg_preds, rat_logits = self.model(features)
            
            loss_cls = self.criterion_cls(cls_logits.squeeze(), labels)
            loss_reg = self.criterion_reg(reg_preds.squeeze(), certainties)
            loss_rat = self.criterion_rat(rat_logits, rationales) # ⭐️ 新增
            
            # ⭐️ 新增：加上 lambda_rat * loss_rat
            loss = loss_cls + (self.lambda_reg * loss_reg) + (self.lambda_rat * loss_rat)
            
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            total_rat_loss += loss_rat.item() # ⭐️ 新增
            
            # 收集预测结果用于计算 Metrics
            cls_probs = torch.sigmoid(cls_logits).squeeze()
            cls_preds = (cls_probs > 0.5).int()
            
            # ⭐️ 新增：处理多标签的 Rationale 预测
            rat_probs = torch.sigmoid(rat_logits)
            rat_preds = (rat_probs > 0.5).int()
            
            all_labels.extend(labels.cpu().numpy())
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_certainties.extend(certainties.cpu().numpy())
            all_reg_preds.extend(reg_preds.squeeze().cpu().numpy())
            
            # ⭐️ 新增：将多维数组展平，方便后续计算 F1 或精确度
            all_rationales.extend(rationales.cpu().numpy().flatten())
            all_rat_preds.extend(rat_preds.cpu().numpy().flatten())

        # 计算评估指标
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_cls_preds)
        f1 = f1_score(all_labels, all_cls_preds, average='macro')
        mae = mean_absolute_error(all_certainties, all_reg_preds)
        pearson_corr, _ = pearsonr(all_certainties, all_reg_preds)
        
        # ⭐️ 新增：计算 Rationale 预测的 Macro F1 分数，看看模型学得准不准
        rat_f1 = f1_score(all_rationales, all_rat_preds, average='macro')
        
        metrics = {
            "val/loss": avg_loss,
            "val/cls_loss": total_cls_loss / len(self.val_loader),
            "val/reg_loss": total_reg_loss / len(self.val_loader),
            "val/rat_loss": total_rat_loss / len(self.val_loader), # ⭐️ 新增
            "val/acc": acc,
            "val/macro_f1": f1,
            "val/mae": mae,
            "val/pearson": pearson_corr,
            "val/rat_macro_f1": rat_f1 # ⭐️ 新增
        }
        
        if wandb.run is not None:
            wandb.log(metrics)
            
        logging.info(f"\nVal Epoch {epoch}: Loss={avg_loss:.4f}, ACC={acc:.4f}, F1={f1:.4f}, MAE={mae:.4f}, Pearson={pearson_corr:.4f}, Rat_F1={rat_f1:.4f}\n")
        return avg_loss, f1

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_f1 = self.eval_epoch(epoch)
            
            # ⭐️ 修改：Early Stopping & Checkpointing 改为基于 val_f1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.early_step = 0
                
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