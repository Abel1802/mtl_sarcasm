import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def create_stratified_datasets(all_dataset, test_size=0.2, random_seed=42):
    # 1. 提取所有样本的标签 (Sarcasm Label)
    logging.info("🔍 Extracting labels for stratified split...")
    targets = [all_dataset[i]['label'].item() for i in range(len(all_dataset))]
    
    # 2. 生成数据集的全局索引列表
    indices = list(range(len(all_dataset)))
    
    # 3. 使用 sklearn 进行严格的分层划分
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=test_size, 
        stratify=targets, 
        random_state=random_seed
    )
    
    # 4. 将划分好的索引重新包装成 PyTorch 可以识别的 Dataset
    train_dataset = Subset(all_dataset, train_indices)
    val_dataset = Subset(all_dataset, val_indices)
    
    # 打印一下划分结果，让你吃颗定心丸
    train_targets = [targets[i] for i in train_indices]
    val_targets = [targets[i] for i in val_indices]
    logging.info(f"✅ Train set: {len(train_dataset)} samples (Positive: {sum(train_targets)}, Negative: {len(train_targets)-sum(train_targets)})")
    logging.info(f"✅ Val set: {len(val_dataset)} samples (Positive: {sum(val_targets)}, Negative: {len(val_targets)-sum(val_targets)})")
    
    return train_dataset, val_dataset


class MultimodalSarcasmDataset(Dataset):
    """
    多模态反讽检测数据集
    负责对齐并加载 Text, Audio, Video 特征以及 Sarcasm, Certainty(强度) 和 Rationale(模态依据) 标签
    """
    def __init__(
        self, 
        csv_path, 
        text_feat_dir, 
        audio_feat_dir, 
        video_feat_dir,
        feat_ext=".pt"
    ):
        super().__init__()
        self.text_feat_dir = text_feat_dir
        self.audio_feat_dir = audio_feat_dir
        self.video_feat_dir = video_feat_dir
        self.feat_ext = feat_ext

        # 0. check directory exists
        assert os.path.exists(csv_path), f"CSV file {csv_path} does not exist!"
        assert os.path.exists(text_feat_dir), f"Text feature directory {text_feat_dir} does not exist!"
        assert os.path.exists(audio_feat_dir), f"Audio feature directory {audio_feat_dir} does not exist!"
        assert os.path.exists(video_feat_dir), f"Video feature directory {video_feat_dir} does not exist!"
        
        # 1. 读取原始标注表
        raw_df = pd.read_csv(csv_path)
        
        # 2. 数据过滤：确保仅保留三种模态特征都存在，且标签合法的样本
        valid_data = []
        missing_count = 0
        
        for _, row in raw_df.iterrows():
            uid = str(row['id'])  
            
            # 构建特征文件路径
            text_path = os.path.join(self.text_feat_dir, f"{uid}{self.feat_ext}")
            audio_path = os.path.join(self.audio_feat_dir, f"{uid}{self.feat_ext}")
            video_path = os.path.join(self.video_feat_dir, f"{uid}{self.feat_ext}")
            
            # 检查特征文件是否全部存在
            if os.path.exists(text_path) and os.path.exists(audio_path) and os.path.exists(video_path):
                
                # 首先确保 sarcasm_label 是有值的
                if pd.notna(row['sarcasm_label']):
                    s_label = float(row['sarcasm_label'])
                    is_valid = False
                    
                    # 动态处理 certainty_1 (强度标签)
                    if pd.notna(row['certainty_1']):
                        c_label = float(row['certainty_1'])
                        is_valid = True
                    elif s_label == 0.0:
                        c_label = 1.0
                        is_valid = True
                    else:
                        is_valid = False
                        
                    # ⭐️ 新增：处理 certainty_2 (模态权重/Rationale标签)
                    # 初始化为全 0 [Text, Audio, Video]
                    rationale = [0.0, 0.0, 0.0] 
                    
                    if is_valid: # 只有前面的逻辑通过了，才继续判断 certainty_2
                        if s_label == 0.0:
                            # 标签为0时，不管它有没有值，都强制设为 [0,0,0]
                            pass 
                        elif s_label == 1.0:
                            # 标签为1时，解析字符串
                            if pd.notna(row['certainty_2']):
                                cert2_str = str(row['certainty_2']).lower()
                                if 't' in cert2_str: rationale[0] = 1.0
                                if 'a' in cert2_str: rationale[1] = 1.0
                                if 'v' in cert2_str: rationale[2] = 1.0
                            else:
                                # 如果是反讽，但缺失 modality 依据，也视为脏数据扔掉
                                is_valid = False
                        
                    # 如果数据合法，加入 valid_data
                    if is_valid:
                        valid_data.append({
                            'id': uid,
                            'label': s_label,
                            'certainty': c_label,
                            'rationale': rationale  # ⭐️ 将新特征存入字典
                        })
                    else:
                        missing_count += 1
                else:
                    missing_count += 1
            else:
                missing_count += 1
                
        self.data = pd.DataFrame(valid_data)
        
        logging.info(f"Loaded dataset from {csv_path}")
        logging.info(f"Total valid samples: {len(self.data)} (Filtered out {missing_count} incomplete samples)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        uid = row['id']
        
        # 1. 动态读取特征
        text_path = os.path.join(self.text_feat_dir, f"{uid}{self.feat_ext}")
        audio_path = os.path.join(self.audio_feat_dir, f"{uid}{self.feat_ext}")
        video_path = os.path.join(self.video_feat_dir, f"{uid}{self.feat_ext}")
        
        text_feat = torch.load(text_path, weights_only=True).squeeze()
        audio_feat = torch.load(audio_path, weights_only=True).squeeze()
        video_feat = torch.load(video_path, weights_only=True).squeeze()
        
        # 2. 获取标签
        label = torch.tensor(row['label'], dtype=torch.float32)
        certainty = torch.tensor(row['certainty'], dtype=torch.float32)
        
        # ⭐️ 新增：将 list 转换为 torch.Tensor
        rationale = torch.tensor(row['rationale'], dtype=torch.float32)
        
        # 3. 组装返回字典
        return {
            'id': uid,
            'text': text_feat,
            'audio': audio_feat,
            'video': video_feat,
            'label': label,
            'certainty': certainty,
            'rationale': rationale
        }

# ==========================================
# 测试与使用示例 (仅作测试用，可注释掉)
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 假设你的目录结构如下，这里演示如何实例化 DataLoader
    dataset = MultimodalSarcasmDataset(
        csv_path="/projects/0/prjs0864/phd_projects/raw_data/sarcasm_zh26/labels_transcriptions_Sheet1_filtered.csv",
        text_feat_dir="/projects/0/prjs0864/phd_projects/mlt_sarcasm/processed_data/text_features",
        audio_feat_dir="/projects/0/prjs0864/phd_projects/mlt_sarcasm/processed_data/audio_features",
        video_feat_dir="/projects/0/prjs0864/phd_projects/mlt_sarcasm/processed_data/video_features",
        feat_ext=".pt"
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 取一个 batch 看看形状
    for batch in dataloader:
        print("Batch Text shape:", batch['text'].shape)       # 期望: [32, 768]
        print("Batch Audio shape:", batch['audio'].shape)     # 期望: [32, 768]
        print("Batch Video shape:", batch['video'].shape)     # 期望: [32, 768]
        print("Batch Labels shape:", batch['label'].shape)    # 期望: [32]
        print("Batch Certainty shape:", batch['certainty'].shape) # 期望: [32]
        break