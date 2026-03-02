import os
import argparse
import json
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.data_loader.data_set import MultimodalSarcasmDataset
from src.data_loader.data_set import create_stratified_datasets
from src.models.mtl_model import GatingMTLModel, CrossAttentionMTLModel
from src.models.collabrative_gate import Speaker_Independent_Triple_Mode_without_Context
from src.trainer.trainer import MTLTrainer

def set_seed(seed: int = 42):
    """固定所有的随机种子，确保实验可绝对复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Sarcasm Detection MTL Training")
    
    # 1. 实验名称与路径配置
    parser.add_argument("--exp_name", type=str, default="baseline_run", 
                        help="Name of the experiment. Creates a subfolder to save logs, configs, and checkpoints.")
    parser.add_argument("--save_dir", type=str, default="output/", 
                        help="Base directory for saving experiments")
    
    # 🌟 新增：随机种子参数
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    
    # 2. 模态消融参数 (Ablation Study)
    parser.add_argument("--ablate_text", action="store_true", help="Ablate (zero out) text modality")
    parser.add_argument("--ablate_audio", action="store_true", help="Ablate (zero out) audio modality")
    parser.add_argument("--ablate_video", action="store_true", help="Ablate (zero out) video modality")

    # 模型与架构
    parser.add_argument("--model_type", type=str, default="gating", choices=["gating", "cross_attn", "collabrative"],
                        help="Choose the fusion model architecture")
    parser.add_argument("--embed_dim", type=int, default=768, help="Feature dimension from extractors")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="Weight for the certainty regression loss")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    
    # 数据集路径配置
    parser.add_argument("--train_csv", type=str, default="raw_data.csv")
    parser.add_argument("--val_csv", type=str, default="data/processed/val.csv")
    parser.add_argument("--text_dir", type=str, default="processed_data/text_features")
    parser.add_argument("--audio_dir", type=str, default="processed_data/audio_features")
    parser.add_argument("--video_dir", type=str, default="processed_data/video_features")
    
    # Wandb 日志控制
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 🌟 新增：在程序一启动，立刻锁定全局随机种子
    set_seed(args.seed)
    
    # ==========================================
    # 📂 实验目录创建与配置保存
    # ==========================================
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 配置日志系统 (双路输出：控制台 + log文件)
    log_file = os.path.join(exp_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 将 args 转化为字典
    config = vars(args)
    config['save_dir'] = exp_dir 
    config['run_name'] = args.exp_name
    config['use_wandb'] = args.disable_wandb
    # config['use_wandb'] = not args.disable_wandb
    config['warmup_ratio'] = 0.1
    config['patience'] = 7  # Early stopping patience
    
    # 将本次实验的超参数配置保存为 json，方便日后复现
    config_save_path = os.path.join(exp_dir, "config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    
    # ==========================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"🌱 Set global random seed to {args.seed}") # 🌟 打印种子信息
    logging.info(f"🚀 Starting training on {device} | Model: {args.model_type} | Exp: {args.exp_name}")
    logging.info(f"📁 Output directory: {exp_dir}")
    logging.info(f"⚙️ Config: {config}")  
    
    # 加载数据 (Load Data)
    logging.info("📦 Loading datasets...")
    all_dataset = MultimodalSarcasmDataset(
        csv_path=args.train_csv,
        text_feat_dir=args.text_dir,
        audio_feat_dir=args.audio_dir,
        video_feat_dir=args.video_dir
    )
    
    # split data_set into train_dataset and val_dataset(0.8:0.2)
    train_dataset, val_dataset = create_stratified_datasets(all_dataset)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 加载模型 (Load Model)
    logging.info(f"🧠 Initializing {args.model_type} model...")
    if args.model_type == "gating":
        model = GatingMTLModel(embed_dim=args.embed_dim)
    elif args.model_type == "cross_attn":
        model = CrossAttentionMTLModel(embed_dim=args.embed_dim)
    elif args.model_type == "collabrative":
        model = Speaker_Independent_Triple_Mode_without_Context()
    else:
        raise ValueError("Invalid model type")

    # 设置优化器 (Optimizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 开始训练 (Start Training)
    logging.info("🔥 Starting training pipeline...")
    trainer = MTLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        device=device
    )
    
    trainer.train()
    logging.info(f"✅ Training completed! All assets saved to {exp_dir}")

if __name__ == "__main__":
    main()