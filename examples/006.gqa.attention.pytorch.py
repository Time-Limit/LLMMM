import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    在MHA中，每个注意力头都有自己独立的Query、Key和Value投影矩阵
    
    公式:
    Attention(Q, K, V) = softmax(Q·K^T/√d_k)·V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)·W^O
    where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # 确保模型维度可以被头数整除
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # 投影矩阵
        self.q_proj = nn.Linear(d_model, d_model)  # Query投影
        self.k_proj = nn.Linear(d_model, d_model)  # Key投影
        self.v_proj = nn.Linear(d_model, d_model)  # Value投影
        self.out_proj = nn.Linear(d_model, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性投影并分割成多个头
        # 形状: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim)
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 转置以便进行注意力计算
        # 形状: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        # 缩放点积注意力: Attention(Q, K, V) = softmax(QK^T/√d_k)V
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获得注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到values
        out = torch.matmul(attn_weights, v)
        
        # 恢复原始形状
        # 形状: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终的线性投影
        out = self.out_proj(out)
        
        return out


class MQA(nn.Module):
    """
    多查询注意力机制 (Multi-Query Attention)
    
    在MQA中，所有头共享相同的Key和Value投影矩阵，但每个头有自己独立的Query投影矩阵
    
    公式与MHA相似，但是Key和Value投影矩阵在所有头之间共享：
    K_shared = K·W^K
    V_shared = V·W^V
    head_i = Attention(Q·W_i^Q, K_shared, V_shared)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # 确保模型维度可以被头数整除
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # Query有多个头，但Key和Value只有一个头
        self.q_proj = nn.Linear(d_model, d_model)  # 多个Query投影
        self.k_proj = nn.Linear(d_model, self.head_dim)  # 单个Key投影
        self.v_proj = nn.Linear(d_model, self.head_dim)  # 单个Value投影
        self.out_proj = nn.Linear(d_model, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_len, kv_len = q.size(1), k.size(1)
        
        # 对Query进行多头投影
        # 形状: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim)
        q = self.q_proj(q).view(batch_size, q_len, self.num_heads, self.head_dim)
        
        # Key和Value只有一个共享投影
        # 形状: (batch_size, seq_len, head_dim)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 转置Query以便进行注意力计算
        # 形状: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        
        # 扩展Key和Value以适应多头格式
        # 形状: (batch_size, 1, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        k = k.unsqueeze(1).expand(batch_size, self.num_heads, kv_len, self.head_dim)
        v = v.unsqueeze(1).expand(batch_size, self.num_heads, kv_len, self.head_dim)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获得注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到values
        out = torch.matmul(attn_weights, v)
        
        # 恢复原始形状
        # 形状: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        # 最终的线性投影
        out = self.out_proj(out)
        
        return out


class GQA(nn.Module):
    """
    分组查询注意力机制 (Grouped-Query Attention)
    
    在GQA中，将注意力头分成几个组，每组共享相同的Key和Value投影矩阵
    
    这是MHA和MQA之间的折中方案：
    - 分组数量 = 1 时，等同于MQA
    - 分组数量 = num_heads 时，等同于MHA
    """
    def __init__(self, d_model, num_heads, num_groups, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头的数量
        self.num_groups = num_groups  # KV组的数量
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # 确保条件满足
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        assert num_heads % num_groups == 0, "num_heads必须能被num_groups整除"
        
        self.heads_per_group = num_heads // num_groups  # 每组的头数量
        
        # 投影矩阵
        self.q_proj = nn.Linear(d_model, d_model)  # Query投影 (所有头)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_groups)  # Key投影 (每组一个)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_groups)  # Value投影 (每组一个)
        self.out_proj = nn.Linear(d_model, d_model)  # 输出投影
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_len, kv_len = q.size(1), k.size(1)
        
        # 对Query进行多头投影
        # 形状: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, head_dim)
        q = self.q_proj(q).view(batch_size, q_len, self.num_heads, self.head_dim)
        
        # 对Key和Value进行分组投影
        # 形状: (batch_size, seq_len, num_groups * head_dim)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 调整Key和Value的形状以适应分组
        # 形状: (batch_size, seq_len, num_groups, head_dim)
        k = k.view(batch_size, kv_len, self.num_groups, self.head_dim)
        v = v.view(batch_size, kv_len, self.num_groups, self.head_dim)
        
        # 转置形状以便计算
        q = q.transpose(1, 2)  # (batch_size, num_heads, q_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_groups, kv_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_groups, kv_len, head_dim)
        
        # 现在需要将q与对应组的k和v匹配
        # 我们需要为每个query头找到对应的key/value组
        
        # 初始化输出张量
        out = torch.zeros(batch_size, self.num_heads, q_len, self.head_dim, device=q.device)
        
        # 对每个组计算注意力
        for i in range(self.num_groups):
            # 获取当前组的queries
            start_idx = i * self.heads_per_group
            end_idx = (i + 1) * self.heads_per_group
            group_q = q[:, start_idx:end_idx]
            
            # 获取当前组的key和value
            group_k = k[:, i:i+1].expand(-1, self.heads_per_group, -1, -1)  # 复制以匹配num_heads
            group_v = v[:, i:i+1].expand(-1, self.heads_per_group, -1, -1)  # 复制以匹配num_heads
            
            # 计算注意力分数
            scores = torch.matmul(group_q, group_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 应用mask（如果提供）
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # 应用softmax获得注意力权重
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 应用注意力权重到values
            out[:, start_idx:end_idx] = torch.matmul(attn_weights, group_v)
        
        # 恢复原始形状
        # 形状: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        # 最终的线性投影
        out = self.out_proj(out)
        
        return out


# 使用示例
def test_attention_mechanisms():
    """测试三种注意力机制的简单示例"""
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    num_groups = 4  # 用于GQA
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 初始化三种注意力机制
    mha = MHA(d_model, num_heads)
    mqa = MQA(d_model, num_heads)
    gqa = GQA(d_model, num_heads, num_groups)
    
    # 前向传播
    mha_output = mha(x, x, x)
    mqa_output = mqa(x, x, x)
    gqa_output = gqa(x, x, x)
    
    print(f"MHA 输出形状: {mha_output.shape}")
    print(f"MQA 输出形状: {mqa_output.shape}")
    print(f"GQA 输出形状: {gqa_output.shape}")
    
    # 比较三种注意力机制的参数数量
    mha_params = sum(p.numel() for p in mha.parameters())
    mqa_params = sum(p.numel() for p in mqa.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())
    
    print(f"MHA 参数数量: {mha_params}")
    print(f"MQA 参数数量: {mqa_params}")
    print(f"GQA 参数数量: {gqa_params}")
    print(f"MQA 相对于 MHA 的参数减少: {1 - mqa_params/mha_params:.2%}")
    print(f"GQA 相对于 MHA 的参数减少: {1 - gqa_params/mha_params:.2%}")


if __name__ == "__main__":
    test_attention_mechanisms()
