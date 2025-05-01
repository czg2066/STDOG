import torch, math
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
    def __init__(self, embed_dim, num_heads, sigma_scales=[1.0, 2.0, 3.0, 4.0, 5.0], alphas=[0.1, 0.2, 0.3, 0.4, 0.5],\
                  num_queries=100):
        super().__init__()
        assert num_heads == len(sigma_scales) == len(alphas), "头数需与尺度参数匹配"
        assert embed_dim%num_heads == 0, "嵌入维度需能被头数整除"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.sigma_scales = sigma_scales
        self.alphas = alphas
        
        # 初始化高斯核缓存
        self.gaussian_kernels = nn.ModuleList([
            self._get_gaussian_kernel(sigma) for sigma in sigma_scales
        ])
        
        # Q/K/V投影
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.pos_encoder = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1) # 位置编码
        self.num_queries = num_queries
        self.queries= nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.proj = nn.Linear(embed_dim*self.num_heads, embed_dim)
        self.out_prof = nn.Linear(num_queries, 12*30)
        self.query_norm = nn.LayerNorm(embed_dim)

        # self.encoder_pos_encoding = PositionEmbeddingSine(self.config.gru_input_size // 2, normalize=True)
        # self.row_embed = nn.Embedding(256, embed_dim)
        # self.col_embed = nn.Embedding(256, embed_dim)

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
        out = F.softmax(out, dim=-1)
        # pos_coss = self.pos_encoder(out)
        # 处理查询
        queries = self.query_norm(self.queries)
        queries = self.queries.repeat(B*self.num_heads, 1, 1)  # [B*num_heads, 50, 1512]
        queries = queries.permute(1, 0, 2) # [50, B*num_heads, 1512]
        # queries = queries.view(B, self.num_heads, self.num_queries, self.embed_dim).permute(2, 0, 1, 3).reshape(50, B*self.num_heads, -1)
        # 处理输入特征
        out = out.permute(1, 0, 2)  # [360, B, 1512]
        out = out.repeat(1, self.num_heads, 1)  # [360, B*num_heads, 1512]
        out, _ = self.cross_attn(
            queries,
            out,
            out
        )
        out = out.permute(1, 0, 2).contiguous().view(B, self.num_queries, -1)
        out = self.proj(out) # [B, self.num_queries, 1512]
        out = self.out_prof(out.permute(0, 2, 1))
        # 恢复原始维度（如果输入是4D）
        if len(orig_shape) == 4:
            _, _, H, W = orig_shape
            out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return out

class TrafficParticipantCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_queries=8, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # 动态参数生成
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        # 空间位置编码
        self.pos_encoder = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.prof = nn.Linear(num_queries, 12*30)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.feat_norm = nn.LayerNorm(embed_dim)
        # 初始化
        nn.init.trunc_normal_(self.query_embed.weight, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_features):
        """
        Args:
            img_features: [B, C, H, W] 图像特征
        Returns:
            traffic_features: [B, num_queries, C] 交通参与者特征
        """
        B, C, H, W = img_features.shape
        # 生成动态查询
        learned_queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]
        learned_queries = self.query_norm(learned_queries)
        learned_queries = self.query_proj(learned_queries)  # [B, num_queries, C]
        # 空间位置增强
        spatial_encoding = self.pos_encoder(img_features)  # [B, C, H, W]
        img_features = img_features + spatial_encoding
        # 展平图像特征
        img_tokens = img_features.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        img_features = self.feat_norm(img_tokens)
        # 交叉注意力
        traffic_features, _ = self.cross_attn(
            query=learned_queries,
            key=img_tokens,
            value=img_tokens,
            need_weights=False
        )
        traffic_features = self.prof(traffic_features.permute(0, 2, 1)) # [B, H*W, C]
        traffic_features = traffic_features.view(B, C, H, W)  # [B, C, H, W]
        return traffic_features


class PositionEmbeddingSine(nn.Module):
  """
  Taken from InterFuser
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError('normalize should be True if scale is passed')
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, tensor):
    x = tensor
    bs, _, h, w = x.shape
    not_mask = torch.ones((bs, h, w), device=x.device)
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if self.normalize:
      eps = 1e-6
      y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
      x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature**(2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos