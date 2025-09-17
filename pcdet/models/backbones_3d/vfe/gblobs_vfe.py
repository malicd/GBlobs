from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
    from torch_geometric.utils import to_dense_batch
except Exception as e:
    # In case someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from .dynamic_voxel_vfe import DynamicVoxelVFE
from ....utils.common_utils import get_voxel_centers


class GBlobsVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.point_cloud_range = kwargs["point_cloud_range"]
        self.voxel_size = kwargs["voxel_size"]

        self.cov_only = model_cfg.get(
            "COVARIANCE_ONLY", False
        )  # only covariance matrix
        self.rel_d = self.model_cfg.get(
            "RELATIVE_DISTANCE", False
        )  # use relative distance instead of absolute

    def get_output_feature_dim(self):
        return 14
        return (
            0 if self.cov_only else self.num_point_features
        ) + self.num_point_features**2  # (rel) position (N) + covariance (NxN)

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = (
            batch_dict["voxels"],
            batch_dict["voxel_num_points"],
        )

        points_mean = voxel_features.sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(
            voxel_features
        )
        points_mean = points_mean / normalizer

        voxel_features, _ = torch.split(voxel_features, [3, 2], dim=2)

        points_mean, features_mean = torch.split(points_mean, [3, 2], dim=1)
        features_mean[:, 0] = 1. # set intensity of ALL datasets to 1 (to match robosense)

        weight = (voxel_features.sum(2, keepdim=True) != 0.0).float()
        cov = torch.einsum(
            "nka,nkb->nab",
            weight * (voxel_features - points_mean.unsqueeze(1)),
            weight * (voxel_features - points_mean.unsqueeze(1)),
        )
        cov = cov / torch.clamp_min(normalizer - 1.0, min=1.0).unsqueeze(dim=-1)

        # assert torch.allclose(cov, cov.permute(0, 2, 1))

        pos_features = points_mean
        if self.rel_d:
            voxel_centers = get_voxel_centers(
                batch_dict["voxel_coords"][:, 1:],
                1,
                self.voxel_size,
                self.point_cloud_range,
            )
            voxel_size = torch.tensor(
                self.voxel_size, device=voxel_centers.device
            ).float()
            pos_features = points_mean[:, 0:3] - voxel_centers

        features = torch.cat([pos_features, cov.reshape(-1, 9), features_mean], dim=1,).contiguous()

        batch_dict["voxel_features"] = features

        return batch_dict


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [
                self.linear(inputs[num_part * self.part : (num_part + 1) * self.part])
                for num_part in range(num_parts + 1)
            ]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = F.gelu(x)
        return x


class DynamicGBlobsVFE(VFETemplate):
    def __init__(
        self,
        model_cfg,
        num_point_features,
        voxel_size,
        grid_size,
        point_cloud_range,
        **kwargs,
    ):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.num_point_features = num_point_features

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [self.num_point_features + self.num_point_features**2] + list(
            self.num_filters
        )

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    self.use_norm,
                    last_layer=(i >= len(num_filters) - 2),
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        if torch.__version__ >= "1.13":  # for stable argument
            self.argsort = partial(torch.argsort, stable=True)
        else:
            self.argsort = partial(torch.argsort)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def _concat_group_by(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        index_count = torch.bincount(index)
        fill_count = index_count.max() - index_count
        fill_zeros = torch.zeros_like(x[0]).repeat(
            fill_count.sum(), *([1] * (len(x.shape) - 1))
        )
        fill_index = (
            torch.arange(0, fill_count.shape[0])
            .to(x.device)
            .repeat_interleave(fill_count)
        )
        index_ = torch.cat([index, fill_index], dim=0)
        x_ = torch.cat([x, fill_zeros], dim=0)
        x_ = x_[self.argsort(index_)].view(
            index_count.shape[0], index_count.max(), *x.shape[1:]
        )
        return x_

    def forward(self, batch_dict, **kwargs):
        points = batch_dict["points"]  # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor(
            (points[:, [1, 2, 3]] - self.point_cloud_range[[0, 1, 2]])
            / self.voxel_size[[0, 1, 2]]
        ).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1, 2]])).all(
            dim=1
        )
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = (
            points[:, 0].int() * self.scale_xyz
            + points_coords[:, 0] * self.scale_yz
            + points_coords[:, 1] * self.scale_z
            + points_coords[:, 2]
        )

        unq_coords, unq_inv, unq_cnt = torch.unique(
            merge_coords, return_inverse=True, return_counts=True, dim=0
        )

        points_mean = torch_scatter.scatter_mean(points[:, 1:], unq_inv, dim=0)

        sorted = torch.argsort(unq_inv)
        voxel_features = to_dense_batch(
            points[sorted, 1:],
            batch=unq_inv[sorted],
            max_num_nodes=self.model_cfg.get("MAX_NUM_POINTS", 32),
        )[0]
        weight = (voxel_features.sum(2, keepdim=True) != 0.0).float()
        cov = torch.einsum(
            "nka,nkb->nab",
            weight * (voxel_features - points_mean.unsqueeze(1)),
            weight * (voxel_features - points_mean.unsqueeze(1)),
        )
        cov = cov / torch.clamp_min(unq_cnt - 1.0, min=1.0).view(-1, 1, 1)
        # assert torch.allclose(cov, cov.permute(0, 2, 1))

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack(
            (
                unq_coords // self.scale_xyz,
                (unq_coords % self.scale_xyz) // self.scale_yz,
                (unq_coords % self.scale_yz) // self.scale_z,
                unq_coords % self.scale_z,
            ),
            dim=1,
        )
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        voxel_centers = get_voxel_centers(
            voxel_coords[:, 1:],
            1,
            self.voxel_size.detach().cpu().numpy(),
            self.point_cloud_range.detach().cpu().numpy(),
        )
        pos_features = points_mean[:, 0:3] - voxel_centers

        features = torch.cat(
            [pos_features, cov.reshape(-1, self.num_point_features**2)],
            dim=1,
        ).contiguous()

        for pfn in self.pfn_layers:
            features = pfn(features)

        batch_dict["pillar_features"] = batch_dict["voxel_features"] = features
        batch_dict["voxel_coords"] = voxel_coords

        return batch_dict
