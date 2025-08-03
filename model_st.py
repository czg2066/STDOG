import torch, timm, cv2
import torch.nn as nn
import torch.nn.functional as F
from Attention import LateralInhibitionAttention, MultiScaleDiffusionAttention, TrafficParticipantCrossAttention
from DSdecoder import PerspectiveDecoder
from STAttention import STTransformerWithDeformableAttention

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.converter_class = 7
        self.bev_converter_class = 11
        self.backbone = timm.create_model(
            'regnety_032',
            pretrained=True,
            features_only=True,
        )
        backbone_channels = self.backbone.feature_info.channels() # [c1, c2, c3, c4, c5]
        
        # 假设我们使用后4个尺度的特征
        input_channels = backbone_channels[-4:] # [c2, c3, c4, c5]
        d_model = 256 # Transformer的维度
        
        # 添加一个 1x1 卷积层来统一通道数
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, d_model, kernel_size=1),
                nn.GroupNorm(32, d_model),
            ) for in_ch in input_channels
        ])
        self.st_transformer = STTransformerWithDeformableAttention(
            d_model=d_model,
            n_heads=4,
            num_layers=2,
            num_agent_queries=50,
            num_map_queries=50,
            num_plan_queries=8,
            n_levels=len(input_channels),
            n_points=8,
            d_ffn=d_model,
            dropout=0.1
        )
        self.bev_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(3, 3), 
                stride=1,
                padding=1,
                bias=True
            ),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(
               in_channels=d_model,
               out_channels=self.bev_converter_class,
               kernel_size=(1, 1),
               stride=1,
               padding=0,
               bias=True
            ),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
        )
        # 上采用query到map大小 
        self.query_to_map = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(d_model // 2, d_model, kernel_size=1)
        )
        self.depth_decoder = PerspectiveDecoder(in_channels=backbone_channels[-1],
                                               out_channels=1,
                                               inter_channel_0=512,
                                               inter_channel_1=128,
                                               inter_channel_2=32,
                                               scale_factor_0=8,
                                               scale_factor_1=4)
        self.semantic_decoder = PerspectiveDecoder(in_channels=backbone_channels[-1],
                                                  out_channels=self.converter_class,
                                                  inter_channel_0=512,
                                                  inter_channel_1=128,
                                                  inter_channel_2=32,
                                                  scale_factor_0=8,
                                                  scale_factor_1=4)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((360, 960))
        
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 2)  # 输出 (x, y)
        )
        # self.top_down = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=d_model,
        #         out_channels=d_model,
        #         kernel_size=(1, 1)
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(
        #         in_channels=d_model,
        #         out_channels=d_model,
        #         kernel_size=(3, 3),
        #         stride=2,
        #         padding=1
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(
        #         in_channels=d_model,
        #         out_channels=d_model,
        #         kernel_size=(3, 3),
        #         stride=2,
        #         padding=1
        #     ),
        #     nn.ReLU(inplace=True)
        # )
        self.bev_converter = [
            0,  # unlabeled
            1,  # road
            2,  # sidewalk
            3,  # lane_markers
            4,  # lane_markers broken, you may cross them
            5,  # stop_signs
            6,  # traffic light green
            7,  # traffic light yellow
            8,  # traffic light red
            9,  # vehicle
            10,  # walker
        ]
        self.bev_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.converter = [
        0,  # unlabeled, static, building, fence, other, pole, dynamic, water, terrain,\
        # wall, traffic sign, sky, ground, bridge, rail track, guard rail, vegetation
        4,  # pedestrian
        5,  # road line
        2,  # road
        6,  # sidewalk
        1,  # vehicle
        3,  # traffic light
        ]   
        self.semantic_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.bev_semantic_loss = nn.CrossEntropyLoss(weight=self.bev_weight)
        self.depth_loss = nn.L1Loss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.semantic_weights)
        self.trajectory_loss = nn.L1Loss()


    def forward(self, x):
        # img_features = []
        # for i in range(6):
        #     feat = self.backbone(x[i])
        #     img_features.append(feat)
        #     del feat
        # front_row = torch.cat([img_features[0], img_features[2], img_features[1]], dim=3)
        # back_row = torch.cat([img_features[3], img_features[5], img_features[4]], dim=3)
        # img_features = torch.cat([front_row, back_row], dim=2)
        # del front_row, back_row
        x = x.contiguous()
        backbone_features = self.backbone(x)[-4:] # 取后4个尺度的特征
        
        # 2. 统一通道数并收集 spatial_shapes
        srcs = []
        spatial_shapes_list = []
        for i, feature in enumerate(backbone_features):
            # (N, C_in, H, W) -> (N, d_model, H, W)
            feature = feature.contiguous()
            proj_feature = self.input_proj[i](feature)
            srcs.append(proj_feature)
            spatial_shapes_list.append(proj_feature.shape[-2:])
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=x.device)
        # 计算每个尺度特征图展平后的起始索引
        # level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 将所有尺度的特征图展平并拼接
        # (N, d_model, H, W) -> (N, H*W, d_model)
        flattened_srcs = [src.flatten(2).permute(0, 2, 1) for src in srcs]
        # 拼接成 (N, sum(H*W), d_model)
        feature_maps_3d = torch.cat(flattened_srcs, dim=1)
        feature_maps_3d = feature_maps_3d.permute(1, 0, 2)
        # img_features = torch.cat(img_features, dim=1)
        agent_query, map_query, plan_query = self.st_transformer(feature_maps_3d, spatial_shapes)
        agent_query_reshaped = agent_query.permute(0, 2, 1)
        N, C, _ = agent_query_reshaped.shape
        H_q, W_q = 5, 10
        agent_query_image = agent_query_reshaped.view(N, C, H_q, W_q)
        # plan_query_reshaped = plan_query.permute(0, 2, 1)
        # N_p, C_p, Lq_p = plan_query_reshaped.shape
        # H_p, W_p = 3, 8
        # plan_query_image = plan_query_reshaped.view(N_p, C_p, H_p, W_p)
        # img_features_td = self.top_down(img_features)
        # img_features = img_features + img_features_td
        semantic_features = self.semantic_decoder(backbone_features[-1])
        semantic_features = self.adaptive_pool(semantic_features)
        depth_features = self.depth_decoder(backbone_features[-1])
        depth_features = self.adaptive_pool(depth_features)
        depth_features = torch.sigmoid(depth_features).squeeze(1)
        map_features = self.query_to_map(agent_query_image)
        bev_features = self.bev_decoder(map_features)
        trajectory_features = self.trajectory_decoder(plan_query)
        return semantic_features, depth_features, bev_features, trajectory_features
    
    def check_tensor_integrity(self, tensor, name):
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} 包含NaN值")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} 包含Inf值")
        if not tensor.is_contiguous():
            print(f"警告: {name} 内存不连续")

    def loss_all(self, semantic_features, depth_features, bev_features, trajectory_features, semantic_labels, depth_labels, bev_labels, ego_waypoints):
        semantic_loss = self.semantic_loss(semantic_features, semantic_labels)
        weight_matrix = self.caclu_map(semantic_labels)
        semantic_loss = semantic_loss * weight_matrix
        semantic_loss = torch.mean(semantic_loss)

        bev_semantic_loss = self.bev_semantic_loss(bev_features, bev_labels)
        # bev_map = self.torch_dog_filter(bev_labels)
        # weight_matrix = 1.0 + (alpha_bsm - 1.0) * bev_map
        weight_matrix = self.caclu_map(bev_labels)
        bev_semantic_loss = bev_semantic_loss * weight_matrix
        bev_semantic_loss = torch.mean(bev_semantic_loss)

        depth_loss = self.depth_loss(depth_features, depth_labels)
        # depth_map = self.torch_dog_filter(depth_labels)
        # weight_matrix = 1.0 + (alpha_dm - 1.0) * depth_map
        weight_matrix = self.caclu_map(depth_labels)
        depth_loss = depth_loss * weight_matrix
        depth_loss = torch.mean(depth_loss)
        trajectory_loss = self.trajectory_loss(trajectory_features, ego_waypoints)
        trajectory_loss = torch.mean(trajectory_loss)
        return [semantic_loss, depth_loss, bev_semantic_loss, trajectory_loss]

    def caclu_map(self, labels, weight=[1.0, 1.0, 1.0, 1.0]):
        semantic_map_tiny = self.torch_dog_filter(labels, sigma1=0.8, sigma2=1.2, kernel_size=5)       #细小物体如车道线的DoG滤波器
        semantic_map_small = self.torch_dog_filter(labels, sigma1=1.2, sigma2=2.0, kernel_size=7)      #小型物体如人体的DoG滤波器
        semantic_map_mid = self.torch_dog_filter(labels, sigma1=1.5, sigma2=3.0, kernel_size=9)        #中型物体如车辆的DoG滤波器
        semantic_map_big = self.torch_dog_filter(labels, sigma1=2.0, sigma2=4.5, kernel_size=13)       #大型物体如卡车的DoG滤波器
        weight_matrix = 1 + weight[0] * semantic_map_tiny + weight[1] * semantic_map_small +\
              weight[2] * semantic_map_mid + weight[3] * semantic_map_big
        return weight_matrix

    def torch_dog_filter(self, tensor, sigma1=1.2, sigma2=2.0, kernel_size=5):
        """
        增强鲁棒性的PyTorch DoG实现
        输入支持:
            - 整型(Long)或浮点型(Float)张量
            - 任意设备(CPU/GPU)
            - 形状: [B, H, W] 或 [B, C, H, W]
        """
        # 类型转换与维度检查
        if tensor.dtype != torch.float32:
            tensor = tensor.float()  # 强制转换为浮点型
            
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)  # [B, 1, H, W]
        elif tensor.dim() != 4:
            raise ValueError(f"Invalid input shape: {tensor.shape}")

        # 高斯核生成（显式指定dtype）
        def _gaussian_kernel(sigma):
            x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size, 
                            device=tensor.device, dtype=tensor.dtype)
            y = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size,
                            device=tensor.device, dtype=tensor.dtype)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            kernel = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
            return kernel / kernel.sum()

        # 生成卷积核
        kernel1 = _gaussian_kernel(sigma1).view(1, 1, kernel_size, kernel_size)
        kernel2 = _gaussian_kernel(sigma2).view(1, 1, kernel_size, kernel_size)
        
        # 卷积操作（保持维度一致）
        pad = kernel_size // 2
        gaussian1 = F.conv2d(tensor, kernel1, padding=pad)
        gaussian2 = F.conv2d(tensor, kernel2, padding=pad)
        
        # 结果处理
        dog = gaussian1 - gaussian2
        return dog.squeeze(1) if dog.shape[1] == 1 else dog  # 恢复原始维度
