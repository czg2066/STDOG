import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention

class STTransformerWithDeformableAttention(nn.Module):
    """
    一个为自动驾驶设计的，处理多种查询的时空可变形Transformer。
    该模块内部定义了agent, map, plan三种可学习的查询，并通过多层网络进行信息交互和融合。
    - Agent分支使用SpatioTemporalAttentionWithRefPoints处理时空特征。
    - Map分支使用torchvision官方的MultiScaleDeformableAttention处理地图特征。
    - Plan分支通过交叉注意力融合Agent和Map的信息，进行轨迹规划。
    """
    def __init__(self, d_model=256, n_heads=8, num_layers=6,
                 num_agent_queries=50, num_map_queries=50, num_plan_queries=5,
                 n_levels=4, n_points=4, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_points = n_points

        # --- 可学习的查询和参考点 ---
        self.agent_query_embed = nn.Embedding(num_agent_queries, d_model)
        self.map_query_embed = nn.Embedding(num_map_queries, d_model)
        self.plan_query_embed = nn.Embedding(num_plan_queries, d_model)

        # 参考点也设为可学习的。map的参考点是2D的，agent是4D的
        self.agent_reference_points = nn.Linear(d_model, n_levels * 2)
        self.map_reference_points = nn.Linear(d_model, 2)
        
        # --- Transformer层 ---
        self.layers = nn.ModuleList([
            STDeformableTransformerLayer(
                d_model=d_model, 
                n_heads=n_heads, 
                n_levels=n_levels, 
                n_points=n_points, 
                d_ffn=d_ffn, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, feature_maps, spatial_shapes):
        """
        前向传播函数。
        :param feature_maps: (N, L, C) 多尺度特征图展平后拼接的结果。L是所有层级像素总数。
        :param spatial_shapes: (n_levels, 2) 每个特征层级的形状 (H, W)。
        :return: (agent_query, map_query, plan_query) 处理后的三种查询。
        """
        # N, _, C = feature_maps.shape
        L, N, C = feature_maps.shape
        
        # 获取初始查询
        agent_query_init = self.agent_query_embed.weight.unsqueeze(0).repeat(N, 1, 1)
        map_query_init = self.map_query_embed.weight.unsqueeze(0).repeat(N, 1, 1)
        plan_query_init = self.plan_query_embed.weight.unsqueeze(0).repeat(N, 1, 1)
        # 将 query 转置为 (num_queries, N, C) 并保持
        agent_query = agent_query_init.permute(1, 0, 2)
        map_query = map_query_init.permute(1, 0, 2)
        plan_query = plan_query_init.permute(1, 0, 2)

        # 计算初始参考点
        # agent参考点: (N, num_queries, n_levels, 2)
        agent_ref_pts = self.agent_reference_points(agent_query).permute(1, 0, 2).view(N, -1, self.n_levels, 2).sigmoid()
        # map参考点: (N, num_queries, 2) -> (N, num_queries, n_levels, 2)
        map_ref_pts_2d = self.map_reference_points(map_query).permute(1, 0, 2).sigmoid()
        map_ref_pts = map_ref_pts_2d.unsqueeze(2).repeat(1, 1, self.n_levels, 1)

        # 逐层通过Transformer
        for layer in self.layers:
            agent_query, map_query, plan_query = layer(
                agent_query, map_query, plan_query,
                feature_maps, spatial_shapes,
                agent_ref_pts, map_ref_pts
            )
        # 在返回之前，可能需要将 query 转置回 (N, num_queries, C)
        agent_query = agent_query.permute(1, 0, 2)
        map_query = map_query.permute(1, 0, 2)
        plan_query = plan_query.permute(1, 0, 2)
        return agent_query, map_query, plan_query


class STDeformableTransformerLayer(nn.Module):
    """
    STTransformerWithDeformableAttention的单层实现。
    此层首先通过一个自注意力模块在所有查询(agent, map, plan)之间进行信息融合，
    然后将融合后的查询送入各自专有的注意力分支进行处理。
    """
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4, d_ffn=1024, dropout=0.1):
        super().__init__()
        # --- Self Attention for all queries (Early Fusion) ---
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.dropout_self_attn = nn.Dropout(dropout)

        # --- Agent Branch ---
        self.agent_attn = SpatioTemporalAttentionWithRefPoints(d_model, n_levels, n_heads, n_points)
        self.agent_ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.agent_norm1 = nn.LayerNorm(d_model)
        self.agent_norm2 = nn.LayerNorm(d_model)
        self.agent_dropout = nn.Dropout(dropout)

        # --- Map Branch ---
        self.map_attn = MultiScaleDeformableAttention(
            embed_dims=d_model, # 确保这里是 embed_dims=d_model
            num_heads=n_heads, 
            num_levels=n_levels, 
            num_points=n_points
        )
        # self.map_attn = SpatioTemporalAttentionWithRefPoints(d_model, n_levels, n_heads, n_points)
        self.map_ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.map_norm1 = nn.LayerNorm(d_model)
        self.map_norm2 = nn.LayerNorm(d_model)
        self.map_dropout = nn.Dropout(dropout)
        
        # --- Plan Branch ---
        self.plan_cross_attn_agent = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.plan_cross_attn_map = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.plan_ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.plan_norm1 = nn.LayerNorm(d_model)
        self.plan_norm2 = nn.LayerNorm(d_model)
        self.plan_norm3 = nn.LayerNorm(d_model)
        self.plan_dropout1 = nn.Dropout(dropout)
        self.plan_dropout2 = nn.Dropout(dropout)

    def forward(self, agent_query, map_query, plan_query, feature_maps, spatial_shapes, agent_ref_pts, map_ref_pts):
        # --- 1. Early fusion via self-attention among all queries ---
        len_agent, len_map, len_plan = agent_query.shape[0], map_query.shape[0], plan_query.shape[0]
        
        # Concatenate, apply self-attention (with pre-norm), and add residual connection
        queries_cat = torch.cat([agent_query, map_query, plan_query], dim=0)
        queries_cat_norm = self.norm_self_attn(queries_cat)
        attn_output, _ = self.self_attn(queries_cat_norm, queries_cat_norm, queries_cat_norm)
        queries_cat = queries_cat + self.dropout_self_attn(attn_output)

        # Split back to individual queries
        agent_query, map_query, plan_query = torch.split(queries_cat, [len_agent, len_map, len_plan], dim=0)

        # --- 2. Specialized attention for each branch ---
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # --- Agent Branch ---
        agent_out = self.agent_attn(self.agent_norm1(agent_query), agent_ref_pts, feature_maps, spatial_shapes)
        agent_query = agent_query + self.agent_dropout(agent_out)
        agent_query = agent_query + self.agent_dropout(self.agent_ffn(self.agent_norm2(agent_query)))

        # --- Map Branch ---
        map_out = self.map_attn(
            query=self.map_norm1(map_query),
            value=feature_maps,
            reference_points=map_ref_pts,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        # map_out = self.map_attn(self.map_norm1(map_query), map_ref_pts, feature_maps, spatial_shapes)
        map_query = map_query + self.map_dropout(map_out)
        map_query = map_query + self.map_dropout(self.map_ffn(self.map_norm2(map_query)))

        # --- Plan Branch ---
        # Cross-attention with agent query
        plan_out, _ = self.plan_cross_attn_agent(self.plan_norm1(plan_query), agent_query, agent_query)
        plan_query = plan_query + self.plan_dropout1(plan_out)
        # Cross-attention with map query
        plan_out, _ = self.plan_cross_attn_map(self.plan_norm2(plan_query), map_query, map_query)
        plan_query = plan_query + self.plan_dropout2(plan_out)
        # FFN
        plan_query = plan_query + self.plan_dropout2(self.plan_ffn(self.plan_norm3(plan_query)))

        return agent_query, map_query, plan_query


class SpatioTemporalAttentionWithRefPoints(nn.Module):
    """
    时空注意力模块，带有参考点。
    这个模块借鉴了 Deformable DETR 的思想，使用可学习的采样偏移量和注意力权重，
    从多尺度特征图中采样和聚合特征。
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        初始化函数。
        :param d_model: 特征维度。
        :param n_levels: 特征图的层级数量。
        :param n_heads: 注意力头的数量。
        :param n_points: 每个查询在每个头和每个层级上的采样点数量。
        """
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 计算每个注意力头的维度
        self.d_head = d_model // n_heads
        # 确认 d_model 可以被 n_heads 整除
        assert self.d_head * n_heads == self.d_model, "d_model must be divisible by n_heads"

        # 线性层，用于从查询生成采样点的偏移量
        self.sampling_offset_net = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 线性层，用于从查询生成注意力权重
        self.attention_weight_net = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 线性层，用于最终的输出投影
        self.output_proj = nn.Linear(d_model, d_model)
        
        # 初始化网络权重
        self._reset_parameters()

    def _reset_parameters(self):
        """
        初始化网络参数。
        """
        # 将采样偏移网络权重初始化为0，这样初始采样点就是参考点
        nn.init.constant_(self.sampling_offset_net.weight, 0)
        # 将采样偏移网络偏置初始化为0
        nn.init.constant_(self.sampling_offset_net.bias, 0)
        # 将注意力权重网络权重初始化为0
        nn.init.constant_(self.attention_weight_net.weight, 0)
        # 将注意力权重网络偏置初始化为0
        nn.init.constant_(self.attention_weight_net.bias, 0)
        # 使用Xavier均匀分布初始化输出投影层权重
        nn.init.xavier_uniform_(self.output_proj.weight)
        # 将输出投影层偏置初始化为0
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, query, reference_points, feature_maps, spatial_shapes):
        """
        前向传播函数。
        :param query: (N, Lq, C) 查询张量，N是批量大小, Lq是查询序列长度, C是特征维度。
        :param reference_points: (N, Lq, n_levels, 2) 参考点坐标，范围在[0, 1]之间。
        :param feature_maps: (N, L, C) 多尺度特征图展平后拼接的结果。L是所有层级像素总数。
        :param spatial_shapes: (n_levels, 2) 每个特征层级的形状 (H, W)。
        :return: (N, Lq, C) 经过注意力加权后的特征。
        """
        Lq, N, C = query.shape
        query_for_sampling = query.permute(1, 0, 2)
        
        # 1. 从查询生成采样偏移量和注意力权重
        # (N, Lq, n_heads * n_levels * n_points * 2)
        sampling_offsets = self.sampling_offset_net(query_for_sampling)
        # (N, Lq, n_heads, n_levels, n_points, 2)
        sampling_offsets = sampling_offsets.view(N, Lq, self.n_heads, self.n_levels, self.n_points, 2)

        # (N, Lq, n_heads * n_levels * n_points)
        attention_weights = self.attention_weight_net(query_for_sampling)
        # (N, Lq, n_heads, n_levels * n_points)
        attention_weights = attention_weights.view(N, Lq, self.n_heads, self.n_levels * self.n_points)
        # 在每个头的每个查询的所有采样点上进行softmax
        attention_weights = F.softmax(attention_weights, -1)
        # (N, Lq, n_heads, n_levels, n_points)
        attention_weights = attention_weights.view(N, Lq, self.n_heads, self.n_levels, self.n_points)

        # 2. 计算采样点坐标
        # (n_levels, 2) -> (1, 1, 1, n_levels, 1, 2)
        spatial_shapes_flipped = spatial_shapes.flip(-1).view(1, 1, 1, self.n_levels, 1, 2)
        # (N, Lq, 1, n_levels, 1, 2)
        reference_points_expanded = reference_points.unsqueeze(2).unsqueeze(4)
        reference_points_expanded = reference_points_expanded.repeat(1, 1, self.n_heads, 1, self.n_points, 1)
        
        # 偏移量归一化到[0, 1]范围
        # (N, Lq, n_heads, n_levels, n_points, 2)
        sampling_offsets = sampling_offsets / spatial_shapes_flipped

        # (N, Lq, n_heads, n_levels, n_points, 2)
        sampling_locations = reference_points_expanded + sampling_offsets
        
        # 3. 准备grid_sample的输入
        # 将特征图从 (N, L, C) 转换回 (N, C, H, W) 的列表
        L_feat, N_feat, C_feat = feature_maps.shape
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        value_list = [feature_maps[level_start_index[i]:level_start_index[i] + H*W, :, :].permute(1, 2, 0).view(N_feat, C_feat, H, W)
                      for i, (H, W) in enumerate(spatial_shapes)]

        # 将采样点坐标从[0, 1]转换到[-1, 1]
        # (N, Lq, n_heads, n_levels, n_points, 2) -> (N * Lq * n_heads, n_levels, n_points, 2)
        sampling_grid = 2 * sampling_locations - 1
        
        # 4. 从多尺度特征图中采样
        sampled_value_list = []
        for lid, (H, W) in enumerate(spatial_shapes):
            # --- 开始修正 ---
            
            # input_feature_map 的形状是 (N, d_model, H, W)
            input_feature_map = value_list[lid]
            
            # grid 的形状是 (N * n_heads, Lq, n_points, 2)
            grid_for_level = sampling_grid[:, :, :, lid, :, :].permute(0, 2, 1, 3, 4).flatten(0, 1)
            # **核心修正**: 将 d_model 分割成 n_heads 个 d_head
            # (N, d_model, H, W) -> (N, n_heads, d_head, H, W)
            input_split_by_head = input_feature_map.view(N, self.n_heads, self.d_head, H, W)
            # (N, n_heads, d_head, H, W) -> (N * n_heads, d_head, H, W)
            input_for_grid_sample = input_split_by_head.flatten(0, 1)
            
            # 现在 input 和 grid 的批次大小都是 N * n_heads，且 input 的通道数是 d_head
            # input: (N * n_heads, d_head, H, W)
            # grid: (N * n_heads, Lq, n_points, 2)
            sampled_value = F.grid_sample(input_for_grid_sample,
                                          grid_for_level, 
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
            # sampled_value 的形状是 (N * n_heads, d_head, Lq, n_points)
            
            # 恢复维度结构
            # (N * n_heads, d_head, Lq, n_points) -> (N, n_heads, d_head, Lq, n_points)
            sampled_value = sampled_value.view(N, self.n_heads, self.d_head, Lq, -1)
            # (N, n_heads, d_head, Lq, n_points) -> (N, Lq, n_heads, d_head, n_points)
            sampled_value = sampled_value.permute(0, 3, 1, 2, 4)
            
            sampled_value_list.append(sampled_value)

        # (N, Lq, n_heads, C, n_levels, n_points)
        sampled_values = torch.stack(sampled_value_list, dim=4)

        # 5. 注意力加权求和
        # (N, Lq, n_heads, n_levels, n_points, 1) * (N, Lq, n_heads, C, n_levels, n_points) -> (N, Lq, n_heads, C, n_levels, n_points)
        output = attention_weights.unsqueeze(3) * sampled_values
        # (N, Lq, n_heads, C)
        output = output.sum(dim=[-1, -2])
        # (N, Lq, C)
        output = output.view(N, Lq, self.d_model)
        
        # 6. 输出投影
        output = self.output_proj(output)

        return output.permute(1, 0, 2)
