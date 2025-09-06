from torch.utils.data import DataLoader
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import config
from datasets.ntu_dataset import NTUDataset


def ntu_dataloader():
    skel_dir = config.ntu_data_dir
    folders = sorted(
        [f for f in os.listdir(skel_dir) if os.path.isdir(os.path.join(skel_dir, f))]
    )

    train_skel_list, train_label_list = [], []
    val_skel_list, val_label_list = [], []
    for folder in folders:
        files = sorted([f for f in os.listdir(os.path.join(skel_dir, folder))])

        for file in files:
            if not file.endswith("norm.npy"):
                continue

            # shape (3, 25, N)
            skel = np.load(os.path.join(skel_dir, folder, file))

            skel = skel[:, [8, 9, 10], :]  # only select arm

            file_name = file.split("_")[0]  # S001C002P001R001A001
            camera_id = file_name[4:8]
            label = file_name[16:]
            if label not in config.NTU_ACTIONS_SUBSET:
                continue

            if camera_id in config.NTU_TRAIN_CAM_ID:
                train_skel_list.append(skel)
                train_label_list.append(label)
            elif camera_id in config.NTU_VAL_CAM_ID:
                val_skel_list.append(skel)
                val_label_list.append(label)

    ntu_train_dataloader = DataLoader(
        NTUDataset(skel_list=train_skel_list, label_list=train_label_list),
        batch_size=config.batch_size,
        shuffle=True,
    )
    ntu_val_dataloader = DataLoader(
        NTUDataset(skel_list=val_skel_list, label_list=val_label_list),
        batch_size=config.batch_size,
        shuffle=False,
    )
    ntu_test_dataloader = None

    return ntu_train_dataloader, ntu_val_dataloader, ntu_test_dataloader


if __name__ == "__main__":
    ntu_train_dataloader, ntu_val_dataloader, ntu_test_dataloader = ntu_dataloader()
