import torch
from torch import nn 
import torch.nn.functional as F 
import math
import json
import numpy as np
import timm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=1)
        self.bn   = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)  # Scale factor (γ) to 1
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=False)
        return x

class ResConvBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, groups=1)
        self.bn1   = nn.BatchNorm2d(n_channels)
        
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, groups=1)
        self.bn2   = nn.BatchNorm2d(n_channels)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)  # Scale factor (γ) to 1
                nn.init.constant_(m.bias, 0.0)
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity 
        x = F.relu(x)
        return x
    
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channel):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channel in in_channels:
            lateral_conv = ConvBlock(in_channel, out_channel, kernel_size=1, padding=0)
            fpn_conv = ConvBlock(out_channel, out_channel)

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        
    def forward(self, features):
        """
            input features:
                (bs * N_cams, C1, H / 4, W / 2)
                (bs * N_cams, C2, H / 8, W / 4)
                (bs * N_cams, C3, H /16, W /16)
                (bs * N_cams, C4, H /32, W /32)
            output features:
                (bs * N_cams, C, H / 4, W / 2)
                (bs * N_cams, C, H / 8, W / 4)
                (bs * N_cams, C, H /16, W /16)
                (bs * N_cams, C, H /32, W /32)
        """
        laterals = []
        for feature, lateral_conv in zip(features, self.lateral_convs):
            lateral = lateral_conv(feature)
            laterals.append(lateral)
        
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear')
        
        outputs = []
        for lateral, fpn_conv in zip(laterals, self.fpn_convs):
            output = fpn_conv(lateral)
            outputs.append(output)
        return outputs
    
class BEVNeck3D(nn.Module):
    def __init__(self, in_channels, out_channels, pre_channels, num_layers=2):
        super().__init__()

        self.fuse_conv = ConvBlock(pre_channels, in_channels, kernel_size=1, padding=0)

        layers = nn.ModuleList()

        layers.append(ResConvBlock(in_channels))
        layers.append(ConvBlock(in_channels, out_channels, stride=1))

        for i in range(num_layers):
            layers.append(ResConvBlock(out_channels))
            layers.append(ConvBlock(out_channels, out_channels))

        self.layers = nn.Sequential(*layers)
    
    def forward(self, bev_feature):
        bev_feature = self.fuse_conv(bev_feature)
        bev_feature = self.layers.forward(bev_feature)
        ## TODO: check the axis order (x, y) or (y, x) ##
        return bev_feature

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(DetectionHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.cls_branch = nn.Conv2d(in_channels=in_channels, out_channels=num_classes*num_anchors, kernel_size=1, stride=1, padding=0)
        self.reg_branch = nn.Conv2d(in_channels=in_channels, out_channels=7*num_anchors, kernel_size=1, stride=1, padding=0)
    
        self.init_weights()

    def init_weights(self):
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        nn.init.normal_(self.reg_branch.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_branch.bias, 0)
        
        nn.init.normal_(self.cls_branch.weight, mean=0, std=0.01)
        # nn.init.constant_(self.cls_branch.bias, 0.01)
        nn.init.constant_(self.cls_branch.bias, -4.59)

    def forward(self, x):
        """
            input: 
                bev_feature: [bs, C, BEV_H, BEV_W]
            output:
                cls_output: [bs, num_anchors * num_classes, BEV_H, BEV_W]
                reg_output: [bs, num_anchors * code_size, BEV_H, BEV_W]
        """
        cls_output = self.cls_branch(x)
        reg_output = self.reg_branch(x)
        return cls_output, reg_output

#BEV-LaneDet
def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod

class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)
    
class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding_offset_y_z, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat)
    
class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance_with_offset_z, self).__init__()

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )
        self.head = InstanceEmbedding_offset_y_z(64, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)
    
class BEVPerceptionModel(nn.Module):
    def __init__(self, n_cams, 
                        projection_path, 
                        cam_order, 
                        backbone_name, 
                        scale_factor, 
                        stride,
                        n_voxels,
                        voxel_size,
                        in_channels,
                        out_channel,
                        sizes,
                        rotations,
                        num_classes,
                        bev_shape):
        super().__init__()
        self.scale_factor = scale_factor
        self.projections = self.get_projection(projection_path, cam_order, self.scale_factor, stride)
        
        self.n_cams = n_cams
        self.extraction_model = timm.create_model(backbone_name, features_only=True, pretrained=True)
        
        self.fpn_neck = FeaturePyramidNetwork(in_channels=in_channels, out_channel=out_channel)
        self.bev_neck = BEVNeck3D(in_channels=64*6, out_channels=256, pre_channels=64*6*4, num_layers=6)
        self.detection = DetectionHead(in_channels=256, num_anchors=len(sizes) * len(rotations), num_classes=num_classes)
        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=256)

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size

    def forward(self, x):
        """
            Input:
                x: image with shape (bs, N_cams, 3, H, W)
            Temp Output:
                feats: list of image features -> volumes: list of BEV feature
                    (bs * N_cams, C1, H / 4, W / 2) -> (bs, C, n_x, n_y, n_z) 
                    (bs * N_cams, C2, H / 8, W / 4) -> (bs, C, n_x, n_y, n_z) 
                    (bs * N_cams, C3, H /16, W /16) -> (bs, C, n_x, n_y, n_z) 
                    (bs * N_cams, C4, H /32, W /32) -> (bs, C, n_x, n_y, n_z) 
                volumes: bev features 
                       (bs, num_levels * C, n_x, n_y, n_z)
                    -> (bs, num_levels * C * n_z, n_x, n_y)
                    
        """
        bs, n_cams, c, h, w = x.size()
        x = x.view(bs * n_cams, c, h, w)

        #### feature extraction ####
        features = self.extraction_model(x)[1:]

        ## FPN Neck ###
        # feat_1 = F.interpolate(features[-1], size=(features[-2].size(2), features[-2].size(3)), mode='bilinear')
        # feats = torch.cat([feat_1, features[-2]], dim=1)
        features = self.fpn_neck(features)
        
        #### 2D to BEV feature map ####
        points = self.get_points(self.n_voxels, self.voxel_size)
        volumes = []
        for feat, projection in zip(features, self.projections):
            _, c, h, w = feat.size()
            volume = self.backproject(feat, points, projection)
            volume = volume.reshape([bs, -1, *self.n_voxels])
            volumes.append(volume)
        volumes = torch.cat(volumes, dim=1)

        bs, c, x, y, z = volumes.shape
        volumes = volumes.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z*c).permute(0, 3, 1, 2)
        
        bev_feat = self.bev_neck(volumes) 
        
        return self.lane_head(bev_feat)
            

    def get_projection(self, projection_path, cam_order, scale_factor, strides):
        """
            Input: 
                Extrinsic and Intrinsic matrix for each camera in 'cam_order' list 
                scale factor: scale of training image compared to original image (default = 0.5)
                strides:      stride of each feature map

            Output:
                projections: lidar to camera matrices for each stride
                    [n_stride, n_cams, 3, 4]
        """
        with open(projection_path, "r") as f:
            projection_list = json.load(f)

        extrinsics, intrinsics = [], []
        for cam in cam_order:
            extrinsics.append(projection_list[cam]['extrinsic'])
            intrinsics.append(projection_list[cam]['intrinsic'])
        extrinsics = torch.from_numpy(np.asarray(extrinsics, dtype=np.float32))
        intrinsics = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32))
        intrinsics[:, :2] *= scale_factor

        projections = []
        for stride in strides:
            projection = []
            for intrinsic, extrinsic in zip(intrinsics, extrinsics):
                intrinsic[:2] /= stride
                projection.append(intrinsic @ extrinsic)
            projection = torch.stack(projection)
            projections.append(projection)
        projections = torch.stack(projections)
        return projections

    @torch.no_grad()
    def get_points(self, n_voxels, voxel_size, origin=[0.5, 0.5, 0]):
        n_voxels = torch.tensor(n_voxels)
        voxel_size = torch.tensor(voxel_size)
        origin = torch.tensor(origin)

        points = torch.stack(
            torch.meshgrid([
                torch.arange(n_voxels[0]), 
                torch.arange(n_voxels[1]), 
                torch.arange(n_voxels[2]),
            ])
        )
        new_origin = origin - n_voxels / 2.0 * voxel_size
        points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
        return points
    
    def backproject(self, features, points, projection):
        '''
            features:   [N_cams, C, H, W]
            points:     [3, N_voxel_x, N_voxel_y, N_voxel_z]
            projection: [N_cams, 3, 4]

            volume:     [C, N_voxel_x, N_voxel_y, N_voxel_z]
        '''
        n_images, n_channels, height, width = features.shape
        _, n_x_voxels, n_y_voxels, n_z_voxels = points.shape
        points = points.view(1, 3, -1).expand(n_images, 3, -1)
        points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)

        points_2d = torch.bmm(projection, points)
        x = (points_2d[:, 0] / points_2d[:, 2]).round().long()
        y = (points_2d[:, 1] / points_2d[:, 2]).round().long()
        z = points_2d[:, 2]
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
        volume = torch.zeros((n_channels, points.shape[-1]), device=features.device)
        # import pdb; pdb.set_trace()
        for i in range(n_images):
            indices = valid[i].nonzero(as_tuple=True)[0]
            # volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
            volume[:, indices] = features[i, :, y[i, indices], x[i, indices]]
        volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
        return volume