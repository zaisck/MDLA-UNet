import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from timm.models.layers import trunc_normal_, LayerNorm
import math
from torch.nn import init
from torch.nn.init import trunc_normal_

class Axial_Dilated_Multi_scale(nn.Module):
    def __init__(self, dim_in, dim_out, d=1, x=8, y=8):
        super(Axial_Dilated_Multi_scale, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.d = d
        
        self.initial_dw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * d, kernel_size=3, padding=1, dilation=1, groups=dim_in),
            nn.GroupNorm(4, dim_in * d),
            nn.GELU()
        )
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in * d, dim_in * d, kernel_size=(1, 5), padding=(0, 2), dilation=1, groups=dim_in * d),
            nn.GroupNorm(4, dim_in * d),
            nn.GELU(),
            nn.Conv2d(dim_in * d, dim_in * d, kernel_size=(5, 1), padding=(2, 0), dilation=1, groups=dim_in * d),
            nn.GroupNorm(4, dim_in * d),
            nn.GELU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in * d, dim_in * d, kernel_size=(1, 7), padding=(0, 3), dilation=1, groups=dim_in * d),  # 改为dilation=1保证尺寸一致
            nn.GroupNorm(4, dim_in * d),
            nn.GELU(),
            nn.Conv2d(dim_in * d, dim_in * d, kernel_size=(7, 1), padding=(3, 0), dilation=1, groups=dim_in * d),  # 改为dilation=1保证尺寸一致
            nn.GroupNorm(4, dim_in * d),
            nn.GELU()
        )
        
        self.branch3_conv = nn.Sequential(
            nn.Conv2d(dim_in * d, dim_in * d, kernel_size=3, padding=2, dilation=2, groups=dim_in * d),  # 保持3x3卷积
            nn.GroupNorm(4, dim_in * d),
            nn.GELU()
        )
        
        reduced_dim = max(dim_in * d // 8, 4)
        self.mixed_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in * d, reduced_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, dim_in * d, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.pos_attention = nn.Conv2d(dim_in * d, 1, 3, padding=1, bias=False)
        
        self.output_conv = nn.Conv2d(dim_in * d, dim_out, 1)
        
        self.residual_proj = None
        if dim_in != dim_out:
            self.residual_proj = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)
            self.residual_bn = nn.GroupNorm(4, dim_out)

    def forward(self, x):
        identity = x
        
        x = self.initial_dw(x)
        
        out1 = self.branch1(x)  # (B, C, H, W)
        out2 = self.branch2(x)  # (B, C, H, W) 
        out3 = self.branch3_conv(x)  # (B, C, H, W)
        
        assert out1.shape == out2.shape == out3.shape, f"Shape mismatch: {out1.shape}, {out2.shape}, {out3.shape}"
        
        multi_scale_out = out1 + out2 + out3
        
        mixed_weight = self.mixed_attention(multi_scale_out)
        multi_scale_out = multi_scale_out * mixed_weight
        
        pos_weight = torch.sigmoid(self.pos_attention(multi_scale_out))
        multi_scale_out = multi_scale_out * pos_weight
        
        multi_scale_out = self.output_conv(multi_scale_out)
        
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
            identity = self.residual_bn(identity)
        
        return multi_scale_out + identity

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x 

class feature_embedding_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7]):
        super().__init__()
        
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        
        input_dim = dim_xl + dim_xl + 1  # xh + xl + mask
        mid_dim = dim_xl // 2

        groups = min(mid_dim, 8)
        if mid_dim % groups != 0:
            for g in range(min(8, mid_dim), 0, -1):
                if mid_dim % g == 0:
                    groups = g
                    break
            else:
                groups = 1 
            
        self.multi_scale_conv = nn.ModuleList([
            nn.Sequential(
                LayerNorm(normalized_shape=mid_dim, data_format='channels_first'),
                nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1,
                         padding=(k_size + (k_size-1) * (d-1)) // 2,
                         dilation=d, groups=groups), 
            ) for d in d_list[:2]
        ])
        
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_dim if d_list[0] != d_list[1] else input_dim, mid_dim * 2, 1),
            nn.GELU()
        )
        
        self.output_proj = nn.Conv2d(mid_dim * 2, dim_xl, 1)
        
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_dim * 2, mid_dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim // 4, mid_dim * 2, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, xh, xl, mask=None):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        
        if mask is not None:
            combined = torch.cat((xh, xl, mask), dim=1)
        else:
            combined = torch.cat((xh, xl), dim=1)
        
        x = self.input_proj(combined)
        x1, x2 = x.chunk(2, dim=1)  
        

        out1 = self.multi_scale_conv[0](x1)
        out2 = self.multi_scale_conv[1](x2)
        
        out = torch.cat([out1, out2], dim=1)
        
        att = self.ca(out)
        out = out * att
        
        out = self.output_proj(out)
        return out
    
class MDLAUNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            Axial_Dilated_Multi_scale(c_list[2], c_list[3], d=1, x=16, y=16),
        )

        self.encoder5 = nn.Sequential(
            Axial_Dilated_Multi_scale(c_list[3], c_list[4], d=1, x=8, y=8),
        )
        self.encoder6 = nn.Sequential(
            Axial_Dilated_Multi_scale(c_list[4], c_list[5], d=1, x=4, y=4),
        )

        if bridge: 
            self.GAB1 = feature_embedding_bridge(c_list[1], c_list[0])
            self.GAB2 = feature_embedding_bridge(c_list[2], c_list[1])
            self.GAB3 = feature_embedding_bridge(c_list[3], c_list[2])
            self.GAB4 = feature_embedding_bridge(c_list[4], c_list[3])
            self.GAB5 = feature_embedding_bridge(c_list[5], c_list[4])
            print('feature_embedding_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')
        
        self.decoder1 = nn.Sequential(
            Axial_Dilated_Multi_scale(c_list[5], c_list[4], d=1, x=4, y=4),
        ) 
        self.decoder2 = nn.Sequential(
            Axial_Dilated_Multi_scale(c_list[4], c_list[3], d=1, x=8, y=8),
        ) 
        self.decoder3 = nn.Sequential(
            Axial_Dilated_Multi_scale(c_list[3], c_list[2], d=1, x=16, y=16),
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, Axial_Dilated_Multi_scale):
            for child in m.modules():
                if isinstance(child, nn.Conv2d):
                    fan_out = child.kernel_size[0] * child.kernel_size[1] * child.out_channels
                    fan_out //= child.groups
                    child.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if child.bias is not None:
                        child.bias.data.zero_()
                elif isinstance(child, nn.GroupNorm):
                    child.weight.data.fill_(1)
                    child.bias.data.zero_()
                elif isinstance(child, nn.Parameter):
                    if 'params_' in str(child):
                        nn.init.ones_(child)

    def forward(self, x):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        t6 = out
        
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        if self.gt_ds: 
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
        else: t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        if self.gt_ds: 
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        else:t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        if self.gt_ds: 
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        else: t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        if self.gt_ds: 
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        else: t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        if self.gt_ds: 
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        else: t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        if self.gt_ds:
            return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2), torch.sigmoid(gt_pre1)), torch.sigmoid(out0)
        else:
            return torch.sigmoid(out0)







