from .detector3d_template import Detector3DTemplate

from pcdet.utils.box_utils import boxes_to_corners_3d
from ..model_utils.model_nms_utils import class_agnostic_nms, class_specific_nms
from ...utils.common_utils import limit_period

from einops import einsum
import torch
import numpy as np

def corners_rect_to_camera(corners):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        corners:  (8, 3) [x0, y0, z0, ...], (x, y, z) is the point coordinate in image rect

    Returns:
        boxes_rect:  (7,) [x, y, z, l, h, w, r] in rect camera coords
    """
    height_group = [(0, 4), (1, 5), (2, 6), (3, 7)]
    width_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    length_group = [(0, 1), (2, 3), (4, 5), (6, 7)]
    height, width, length = 0., 0., 0.
    for index_h, index_w, index_l in zip(height_group, width_group, length_group):
        height += np.linalg.norm(corners[index_h[0], :] - corners[index_h[1], :])
        width += np.linalg.norm(corners[index_w[0], :] - corners[index_w[1], :])
        length += np.linalg.norm(corners[index_l[0], :] - corners[index_l[1], :])

    height, width, length = height*1.0/4, width*1.0/4, length*1.0/4

    vector = corners[0] - corners[5]
    rotation_y = -np.arctan2(vector[2], vector[0])

    center_point = corners.mean(axis=0)
    camera_rect = np.concatenate([center_point, np.array([length, width, height, rotation_y])])

    return camera_rect


class TransFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        if not self.model_cfg.get("TTA", False) or self.training:
            return self._post_processing(batch_dict)

        post_process_cfg = self.model_cfg.POST_PROCESSING
        final_pred_dict = batch_dict["final_box_dicts"]

        points = batch_dict["points"]
        for b in range(batch_dict["batch_size"]):
            bidx = points[:, 0] == b
            bpoints = points[bidx, 1:4]
            bpoints_homo = torch.concatenate(
                [bpoints, torch.ones_like(bpoints[:, 0:1])], dim=1
            )
            lidar_aug_matrix = torch.inverse(batch_dict["lidar_aug_matrix"][b])
            res = lidar_aug_matrix.matmul(bpoints_homo.transpose(1, 0)).transpose(1, 0)
            points[bidx, 1:4] = res[:, 0:3] / res[:, 3:]
        batch_dict["points"] = points

        stacked_boxes = torch.cat([pd["pred_boxes"] for pd in final_pred_dict])

        # stacked_boxes[..., -1] = -stacked_boxes[..., -1] + np.pi / 2.
        # stacked_boxes[..., [3, 4, 5]] = stacked_boxes[..., [4, 5, 3]]

        corners = boxes_to_corners_3d(stacked_boxes)
        # transform corners
        homogeneous_corners = torch.cat(
            [corners, torch.ones_like(corners[..., 0:1])], dim=2
        )

        stacked_aug_matrix = torch.cat(
            [
                torch.inverse(batch_dict["lidar_aug_matrix"][i])[None].expand(
                    self.model_cfg.DENSE_HEAD.NUM_PROPOSALS, -1, -1
                )
                for i in range(batch_dict["batch_size"])
            ]
        )

        homogeneous_transformed_corners = einsum(
            stacked_aug_matrix.unsqueeze(dim=1),
            homogeneous_corners,
            "b n i j, b n j -> b n i",
        )
        homogeneous_transformed_corners = (
            homogeneous_transformed_corners[..., :3]
            / homogeneous_transformed_corners[..., 3:]
        )

        boxes = []
        for corner in homogeneous_transformed_corners.detach().cpu().numpy():
            cs = corners_rect_to_camera(corner)
            boxes.append(cs)
        stacked_scores = torch.cat([pd["pred_scores"] for pd in final_pred_dict])
        stacked_labels = torch.cat([pd["pred_labels"] for pd in final_pred_dict])
        boxes = np.stack(boxes)[:, [0, 1, 2, 4, 3, 5, 6]]
        boxes[..., -1:] = limit_period(-boxes[..., -1:] - np.pi / 2)
        boxes = torch.from_numpy(boxes).to(
            stacked_scores.device
        )

        # remove self-predictions
        keep_idx = torch.linalg.norm(boxes[:, 0:2], dim=1) > 1.
        boxes = boxes[keep_idx]
        stacked_scores = stacked_scores[keep_idx]
        stacked_labels = stacked_labels[keep_idx]

        keep_idx, _ = class_specific_nms(
            stacked_scores,
            boxes,
            stacked_labels - 1.,
            post_process_cfg.NMS_CONFIG,
            score_thresh=post_process_cfg.NMS_CONFIG.SCORE_THRESH,
        )
        final_pred_dict = [
            dict(
                pred_boxes=boxes[keep_idx],
                pred_scores=stacked_scores[keep_idx],
                pred_labels=stacked_labels[keep_idx],
            )
        ]

        _, sorted_idx = torch.sort(final_pred_dict[0]['pred_scores'], descending=True)
        final_pred_dict = [
            dict(
                pred_boxes=final_pred_dict[0]['pred_boxes'][sorted_idx],
                pred_scores=final_pred_dict[0]['pred_scores'][sorted_idx],
                pred_labels=final_pred_dict[0]['pred_labels'][sorted_idx],
            )
        ]

        recall_dict = {}
        recall_dict = self.generate_recall_record(
            box_preds=final_pred_dict[0]['pred_boxes'],
            recall_dict=recall_dict,
            batch_index=0,
            data_dict=batch_dict,
            thresh_list=post_process_cfg.RECALL_THRESH_LIST,
        )

        return final_pred_dict, recall_dict

    def _post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
