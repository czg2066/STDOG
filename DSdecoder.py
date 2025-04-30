import torch
import torch.nn as nn
import torch.nn.functional as F
class PerspectiveDecoder(nn.Module):
  """
  Decodes a low resolution perspective grid to a full resolution output. E.g. semantic segmentation, depth
  """

  def __init__(self, in_channels, out_channels, inter_channel_0, inter_channel_1, inter_channel_2, scale_factor_0,
               scale_factor_1):
    super().__init__()
    self.scale_factor_0 = scale_factor_0
    self.scale_factor_1 = scale_factor_1

    self.deconv1 = nn.Sequential(
        nn.Conv2d(in_channels, inter_channel_0, 3, 1, 1),
        nn.ReLU(True),
        nn.Conv2d(inter_channel_0, inter_channel_1, 3, 1, 1),
        nn.ReLU(True),
    )
    self.deconv2 = nn.Sequential(
        nn.Conv2d(inter_channel_1, inter_channel_2, 3, 1, 1),
        nn.ReLU(True),
        nn.Conv2d(inter_channel_2, inter_channel_2, 3, 1, 1),
        nn.ReLU(True),
    )
    self.deconv3 = nn.Sequential(
        nn.Conv2d(inter_channel_2, inter_channel_2, 3, 1, 1),
        nn.ReLU(True),
        nn.Conv2d(inter_channel_2, out_channels, 3, 1, 1),
    )

  def forward(self, x):
    x = self.deconv1(x)
    x = F.interpolate(x, scale_factor=self.scale_factor_0, mode='bilinear', align_corners=False)
    x = self.deconv2(x)
    x = F.interpolate(x, scale_factor=self.scale_factor_1, mode='bilinear', align_corners=False)
    x = self.deconv3(x)

    return x