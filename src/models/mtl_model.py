import torch
import torch.nn as nn

# ==========================================
# 共享的多任务预测头 (Sarcasm & Certainty)
# ==========================================
class MultiTaskHeads(nn.Module):
    """
    统一的预测模块：接收融合后的特征，分流出分类和回归两个头。
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        # 共享特征提取层
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 主任务：反讽分类 (输出1维，配合 Trainer 中的 BCEWithLogitsLoss)
        self.cls_head = nn.Linear(hidden_dim, 1)
        
        # 辅助任务：确定性回归 (输出1维，配合 Trainer 中的 MSELoss)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        shared_feat = self.shared_mlp(x)
        cls_logits = self.cls_head(shared_feat)
        reg_preds = self.reg_head(shared_feat)
        return cls_logits, reg_preds

# ==========================================
# 方案 A：特定模态门控网络 (Modality-Specific Gating)
# ==========================================
class GatingMTLModel(nn.Module):
    """
    基于门控机制的融合模型。
    通过全局上下文动态抑制或放大各个模态的特征。
    """
    def __init__(self, embed_dim=768, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 门控生成网络：输入为 [当前模态特征 ; 全局特征]，输出维度与当前模态一致
        gate_input_dim = embed_dim + (embed_dim * 3)
        
        self.gate_T = nn.Sequential(nn.Linear(gate_input_dim, embed_dim), nn.Sigmoid())
        self.gate_A = nn.Sequential(nn.Linear(gate_input_dim, embed_dim), nn.Sigmoid())
        self.gate_V = nn.Sequential(nn.Linear(gate_input_dim, embed_dim), nn.Sigmoid())
        
        # 多任务预测头
        self.heads = MultiTaskHeads(input_dim=embed_dim * 3, dropout=dropout)

    def forward(self, features):
        T = features['text']   # [B, 768]
        A = features['audio']  # [B, 768]
        V = features['video']  # [B, 768]
        
        # 1. 构建全局上下文特征
        global_feat = torch.cat([T, A, V], dim=-1) # [B, 2304]
        
        # 2. 计算各个模态的门控权重 (Gate)
        # 数学表达: g_T = Sigmoid(W * [T; global_feat])
        g_T = self.gate_T(torch.cat([T, global_feat], dim=-1))
        g_A = self.gate_A(torch.cat([A, global_feat], dim=-1))
        g_V = self.gate_V(torch.cat([V, global_feat], dim=-1))
        
        # 3. 特征加权 (Element-wise multiplication)
        T_gated = T * g_T
        A_gated = A * g_A
        V_gated = V * g_V
        
        # 4. 融合与预测
        fused_feat = torch.cat([T_gated, A_gated, V_gated], dim=-1)
        cls_logits, reg_preds = self.heads(fused_feat)
        
        return cls_logits, reg_preds

# ==========================================
# 方案 B：跨模态注意力网络 (Cross-Modal Attention)
# ==========================================
class CrossAttentionMTLModel(nn.Module):
    """
    基于文本锚点的跨模态注意力融合模型。
    利用文本作为 Query，去音频和视频中寻找不一致的“反讽线索”。
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 跨模态注意力层 (batch_first=True 非常重要，匹配 [B, SeqLen, Dim])
        # Text -> Audio
        self.cross_attn_TA = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Text -> Video
        self.cross_attn_TV = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer Normalization 增强稳定性
        self.norm_TA = nn.LayerNorm(embed_dim)
        self.norm_TV = nn.LayerNorm(embed_dim)
        
        # 多任务预测头 (融合后的维度: Text原始 + TA特征 + TV特征)
        self.heads = MultiTaskHeads(input_dim=embed_dim * 3, dropout=dropout)

    def forward(self, features):
        # 原始特征 [B, 768]
        T = features['text']
        A = features['audio']
        V = features['video']
        
        # 1. 增加序列维度 [B, SeqLen=1, 768]
        T_seq = T.unsqueeze(1)
        A_seq = A.unsqueeze(1)
        V_seq = V.unsqueeze(1)
        
        # 2. 跨模态注意力计算
        # 以 Text 为 Query 去 Audio 中寻找关注点
        attn_TA, _ = self.cross_attn_TA(query=T_seq, key=A_seq, value=A_seq)
        attn_TA = self.norm_TA(attn_TA + T_seq) # 残差连接 + LayerNorm
        
        # 以 Text 为 Query 去 Video 中寻找关注点
        attn_TV, _ = self.cross_attn_TV(query=T_seq, key=V_seq, value=V_seq)
        attn_TV = self.norm_TV(attn_TV + T_seq)
        
        # 3. 移除序列维度还原回 [B, 768]
        attn_TA = attn_TA.squeeze(1)
        attn_TV = attn_TV.squeeze(1)
        
        # 4. 融合 (以文本为主导，拼接其在音视频中找出的线索)
        fused_feat = torch.cat([T, attn_TA, attn_TV], dim=-1)
        
        # 5. 双任务预测
        cls_logits, reg_preds = self.heads(fused_feat)
        
        return cls_logits, reg_preds