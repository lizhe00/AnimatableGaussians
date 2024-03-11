import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_quaternion

from network.mlp import MLPLinear
from utils.embedder import get_embedder


class HandAvatar(nn.Module):
    def __init__(self,
                 multires = 4,
                 view_multires = -1,
                 pose_dim = 15*4):
        super(HandAvatar, self).__init__()
        self.pos_embedder, self.pos_dim = get_embedder(multires, 3)
        if view_multires == -1:
            self.view_embedder, self.view_dim = None, 0
        else:
            self.view_embedder, self.view_dim = get_embedder(view_multires, 3)
        self.pose_dim = pose_dim
        self.tex_mlp = MLPLinear(
            in_channels = self.pos_dim + 1 + self.view_dim + pose_dim,
            inter_channels = [64, 64, 64, 64, 64],
            out_channels = 3,
            last_op = nn.Sigmoid()
        )

    def forward(self, cano_xyz, sdf, view_dir, hand_pose):
        batch_size, n_pts = cano_xyz.shape[:2]
        in_feat = torch.cat([self.pos_embedder(cano_xyz), sdf], -1)
        hand_pose = axis_angle_to_quaternion(hand_pose.reshape(batch_size, -1, 3)).reshape(batch_size, -1)
        if self.view_embedder is not None:
            in_feat = torch.cat([in_feat, self.view_embedder(view_dir)], -1)
        in_feat = torch.cat([in_feat, hand_pose[:, None].expand(-1, n_pts, -1)], -1)
        color = self.tex_mlp(in_feat)
        return color
