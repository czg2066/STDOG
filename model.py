import torch, timm, cv2
import torch.nn as nn
import torch.nn.functional as F
from Attention import LateralInhibitionAttention, MultiScaleDiffusionAttention
from DSdecoder import PerspectiveDecoder

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
        backone_channels = self.backbone.feature_info.channels()[-1]
        self.LI_attn = LateralInhibitionAttention(
            embed_dim=backone_channels,
            num_heads=3,
            beta=0.2,
            neighbor_size=3
        )
        self.MSD_attn = MultiScaleDiffusionAttention(
            embed_dim=backone_channels,
            num_heads=6,
            sigma_scales=[1.0, 2.0, 2.5, 3.0, 4.0, 5.0],
            alphas=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
        )
        self.bev_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=backone_channels,
                out_channels=backone_channels,
                kernel_size=(3, 3), 
                stride=1,
                padding=1,
                bias=True
            ),
            nn.BatchNorm2d(backone_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
               in_channels=backone_channels,
               out_channels=self.bev_converter_class,
               kernel_size=(1, 1),
               stride=1,
               padding=0,
               bias=True
            ),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
        )
        self.depth_decoder = PerspectiveDecoder(in_channels=backone_channels,
                                               out_channels=1,
                                               inter_channel_0=512,
                                               inter_channel_1=128,
                                               inter_channel_2=32,
                                               scale_factor_0=8,
                                               scale_factor_1=4)
        self.semantic_decoder = PerspectiveDecoder(in_channels=backone_channels,
                                                  out_channels=self.converter_class,
                                                  inter_channel_0=512,
                                                  inter_channel_1=128,
                                                  inter_channel_2=32,
                                                  scale_factor_0=8,
                                                  scale_factor_1=4)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((360, 960))
        self.top_down = nn.Sequential(
            nn.Conv2d(
                in_channels=backone_channels,
                out_channels=backone_channels,
                kernel_size=(1, 1)
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(
                in_channels=backone_channels,
                out_channels=backone_channels,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(
                in_channels=backone_channels,
                out_channels=backone_channels,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )
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
        self.bev_weight = torch.tensor([1.0, 1.2, 1.0, 1.2, 1.2, 1.0, 1.1, 1.1, 1.1, 1.2, 1.2])
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
        self.semantic_weights = torch.tensor([1.0, 1.2, 1.0, 1.1, 1.0, 1.1, 1.0])
        self.bev_semantic_loss = nn.CrossEntropyLoss(weight=self.bev_weight)
        self.depth_loss = nn.L1Loss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.semantic_weights)


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
        img_features = self.backbone(x)
        img_features = img_features[-1]
        img_features_td = self.top_down(img_features)
        img_features = img_features + img_features_td
        img_features_LIattn = self.LI_attn(img_features)
        img_features = img_features + img_features_LIattn
        img_features_MSDattn = self.MSD_attn(img_features)
        img_features = img_features + img_features_MSDattn
        semantic_features = self.semantic_decoder(img_features)
        semantic_features = self.adaptive_pool(semantic_features)
        depth_features = self.depth_decoder(img_features)
        depth_features = self.adaptive_pool(depth_features)
        depth_features = torch.sigmoid(depth_features).squeeze(1)
        bev_features = self.bev_decoder(img_features)
        return semantic_features, depth_features, bev_features
    
    def check_tensor_integrity(self, tensor, name):
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} 包含NaN值")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} 包含Inf值")
        if not tensor.is_contiguous():
            print(f"警告: {name} 内存不连续")

    def loss_all(self, semantic_features, depth_features, bev_features, semantic_labels, depth_labels, bev_labels):
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
        return [semantic_loss, depth_loss, bev_semantic_loss]

    def caclu_map(self, labels, weight=[0.8, 1.0, 1.2, 0.8]):
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
