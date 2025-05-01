import torch
import torch.nn as nn
import torch.nn.functional as F

class LateralInhibitionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, beta=0.2, neighbor_size=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.beta = beta
        self.neighbor_size = neighbor_size
        
        # 定义Q/K/V投影
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        
        # 邻域抑制卷积核（模拟周围位置的抑制）
        self.inhibition_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=neighbor_size,
            padding=neighbor_size//2,
            bias=False
        )
        # # 初始化卷积核权重为全1（用于求和邻域）
        # nn.init.constant_(self.inhibition_conv.weight, 1.0)
        # self.inhibition_conv.weight.requires_grad = False  # 固定权重

        # 初始化卷积核：中心为0，周围为1
        ####v9####
        kernel = torch.ones(neighbor_size, neighbor_size)
        kernel[neighbor_size//2, neighbor_size//2] = 0  # 中心位置置零
        self.inhibition_conv.weight.data = kernel.view(1, 1, neighbor_size, neighbor_size)
        self.inhibition_conv.weight.requires_grad = False
        self.norm = nn.LayerNorm(embed_dim)
        #end

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 4:
            # 输入为CNN特征图时 [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x = x.view(B, C, H*W).permute(0, 2, 1)
        B, N, C = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        # 计算原始注意力得分
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, num_heads, N, N]
        # 计算抑制项：对每个位置的邻域求和
        # 将attn视为图像，使用卷积操作求和邻域
        batch_heads = B * self.num_heads
        attn_2d = attn.view(batch_heads, 1, N, N)  # 合并B和num_heads维度
        neighbor_sum = self.inhibition_conv(attn_2d).view(B, self.num_heads, N, N)
        # 侧抑制公式：原始得分 - beta * 邻域总和
        attn = attn - self.beta * neighbor_sum
        ####v9####
        attn = attn - attn.max(dim=-1, keepdim=True).values  # 增加数值稳定性
        #end
        # 标准化与加权输出
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 恢复原始维度（如果输入是4D）
        if len(orig_shape) == 4:
            _, _, H, W = orig_shape
            out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class MultiScaleDiffusionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sigma_scales=[1.0, 2.0, 3.0, 4.0, 5.0], alphas=[0.1, 0.2, 0.3, 0.4, 0.5]):
        super().__init__()
        assert num_heads == len(sigma_scales) == len(alphas), "头数需与尺度参数匹配"
        assert embed_dim%num_heads == 0, "嵌入维度需能被头数整除"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sigma_scales = sigma_scales
        self.alphas = alphas
        
        # 初始化高斯核缓存
        self.gaussian_kernels = nn.ModuleList([
            self._get_gaussian_kernel(sigma) for sigma in sigma_scales
        ])
        
        # Q/K/V投影
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def _get_gaussian_kernel(self, sigma, kernel_size=7):
        """生成2D高斯卷积核"""
        x = torch.arange(-kernel_size//2+1, kernel_size//2+1, dtype=torch.float)
        conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False, padding_mode='reflect')
        gauss = torch.exp(-x**2 / (2*sigma**2))
        # kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)  # 外积
        ####v9####
        kernel = torch.outer(gauss, gauss)
        kernel /= kernel.sum()
        with torch.no_grad():
            conv.weight.data = kernel.view(1,1,kernel_size,kernel_size)
        conv.weight.requires_grad_(False) # 固定权重
        #end
        return conv

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 4:
            # 输入为CNN特征图时 [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = x.shape
            x = x.view(B, C, H*W).permute(0, 2, 1)
        B, N, C = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        # 计算基础注意力得分
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, h, N, N]
        # 对每个头应用不同尺度的高斯扩散
        head_outputs = []
        for h in range(self.num_heads):
            attn_head = attn[:, h, :, :].unsqueeze(1)  # [B, 1, N, N]
            # 应用高斯模糊（多尺度扩散）
            blurred = self.gaussian_kernels[h](attn_head)  # [B, 1, N, N]
            # 结合原始得分与扩散项
            combined = attn_head + self.alphas[h] * blurred
            combined = F.softmax(combined, dim=-1)
            # 计算头输出
            head_out = (combined @ v[:, h, :, :].unsqueeze(1))
            head_outputs.append(head_out)
        # 合并多头结果
        out = torch.cat(head_outputs, dim=1).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        # 恢复原始维度（如果输入是4D）
        if len(orig_shape) == 4:
            _, _, H, W = orig_shape
            out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class TrafficQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_queries=20, num_heads=8, H=12, W=30):
        super().__init__()
        self.num_queries = num_queries
        self.query_embeds = nn.Parameter(torch.randn(num_queries, embed_dim))
        # 注意力机制
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # 动态位置编码
        self.pos_encoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        )
        self.proj = nn.Linear(self.num_queries, H*W, bias=False)

    def forward(self, img_features):
        """img_features: [B, C, H, W]"""
        B, C, H, W = img_features.shape
        # 生成位置编码
        pos_emb = self.pos_encoder(img_features)  # [B, C, H, W]
        pos_emb = pos_emb.view(B, C, H*W).permute(2, 0, 1)  # [H*W, B, C]
        # 初始化查询
        queries = self.query_embeds.unsqueeze(1).repeat(1, B, 1)  # [Q, B, C]
        # 自注意力更新查询
        queries, _ = self.self_attn(queries, queries, queries)  # 自适应的参与者关系建模
        # 交叉注意力聚合图像特征
        img_tokens = img_features.view(B, C, -1).permute(2, 0, 1)  # [H*W, B, C]
        context, _ = self.cross_attn(
            query=queries,
            key=img_tokens + pos_emb,
            value=img_tokens
        )# [Q, B, C]
        context = self.proj(context.permute(1, 2, 0)).view(B, C, H, W)  # [B, C, H, W]
        return context
