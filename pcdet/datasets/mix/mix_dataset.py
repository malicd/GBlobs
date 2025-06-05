from pathlib import Path

import numpy as np

from ..dataset import DatasetTemplate
from ..waymo.waymo_dataset import WaymoDataset
from ..nuscenes.nuscenes_dataset import NuScenesDataset
from ..once.once_dataset import ONCEDataset
from ..kitti.kitti_dataset import KittiDataset


class MixDataset(DatasetTemplate):
    SUPPORTED_DATASETS = {
        "WaymoDataset": WaymoDataset,
        "NuScenesDataset": NuScenesDataset,
        "ONCEDataset": ONCEDataset,
        "KittiDataset": KittiDataset,
    }

    def __init__(
        self, dataset_cfg, class_names, training=True, root_path=None, logger=None
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )

        # Load all datasets

        self.datasets = []

        for ds_idx in range(10):
            cur_ds_cfg = dataset_cfg.get(f"DATASET_{ds_idx}", None)
            if cur_ds_cfg is None:
                break
            if cur_ds_cfg.DATASET in dataset_cfg.get("IGNORE_DATASET", []):
                continue
            # share POINT_CLOUD_RANGE and DATA_PROCESSOR
            cur_ds_cfg.POINT_CLOUD_RANGE = dataset_cfg.POINT_CLOUD_RANGE
            cur_ds_cfg.DATA_PROCESSOR = dataset_cfg.DATA_PROCESSOR

            self.datasets.append(
                self.SUPPORTED_DATASETS[cur_ds_cfg.DATASET](
                    dataset_cfg=cur_ds_cfg,
                    class_names=cur_ds_cfg.CLASS_NAMES,
                    root_path=Path(cur_ds_cfg.DATA_PATH),
                    training=training,
                    logger=logger,
                )
            )

        if dataset_cfg.get("RESAMPLE_ALL_TO_SMALLEST", True):
            # Resample all datasets to the size of the smallest
            self.min_len = np.min([ds.__len__() for ds in self.datasets])
            for ds_idx in range(self.datasets.__len__()):
                sel_idx = np.random.choice(
                    self.datasets[ds_idx].infos.__len__(),
                    size=self.min_len,
                    replace=False,
                )
                self.datasets[ds_idx].infos = [
                    self.datasets[ds_idx].infos[si] for si in sel_idx
                ]

        self.index_map = []
        for ds_idx in range(self.datasets.__len__()):
            for s_idx in range(self.datasets[ds_idx].infos.__len__()):
                self.index_map.append((ds_idx, s_idx))

        assert self.index_map.__len__() == self.__len__()

        if self.logger is not None:
            self.logger.info("Total samples for Mix dataset: %d" % (self.__len__()))

    def __len__(self):
        return np.sum([ds.__len__() for ds in self.datasets])

    def __getitem__(self, index):
        ds_idx, s_idx = self.index_map[index]

        return self.datasets[ds_idx].__getitem__(s_idx)

    def evaluation(self, det_annos, class_names, **kwargs):
        assert False, "implement"
