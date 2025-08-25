from torch.utils.data import DataLoader
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import config
from datasets.mhad_dataset import MHADDataset

def mhad_dataloader(mode=['train', 'val', 'test']):
    mhad_train_dataloader, mhad_val_dataloader, mhad_test_dataloader = None, None, None

    for m in mode:
        accel_list, skel_list, label_list = [], [], []

        subject_ids = config.MHAD_TRAIN_S_IDS if m == 'train' else (
            config.MHAD_VAL_S_IDS if m == 'val' else config.MHAD_TEST_S_IDS)

        for s in subject_ids:
            s_dir = os.path.join(config.mhad_data_dir, s)

            files = sorted(os.listdir(s_dir))

            # group files by 4 a*_s*_t*{intertial_std.npy, inertial.npy, skeleton_upsample_normal.npy, skeleton.npy}
            file_groups = [files[i:i+4] for i in range(0, len(files), 4)]

            for group in file_groups:
                '''
                group = ['a?_s?_t?_inertial.npy', 'a?_s?_t?_inertial_std.npy', 'a?_s?_t?_skeleton.npy', 'a?_s?_t?_skeleton_upsample.npy']
                '''
                label = group[0].split('_')[0]
                if label not in config.mhad_actions_subset:
                    continue

                accel_file = group[1]  # 'a?_s?_t?_inertial_std.npy'
                accel = np.load(os.path.join(s_dir, accel_file)
                                ).astype(config.np_dtype)  # (3, N)

                skel_file = group[3]  # 'a?_s?_t?_skeleton_upsample.npy'
                skel = np.load(os.path.join(s_dir, skel_file)).astype(
                    config.np_dtype)  # (3, joints, N)
                # right hand: shoulder, elbow, wrist
                skel = skel[:, [8, 9, 10], :]  # (3, 3, N)

                accel_list.append(accel)
                skel_list.append(skel)
                label_list.append(label)

        mhad_dataset = MHADDataset(accel_list, skel_list, label_list)
        if m == 'train':
            mhad_train_dataloader = DataLoader(
                mhad_dataset, batch_size=config.batch_size, shuffle=True)
        elif m == 'val':
            mhad_val_dataloader = DataLoader(
                mhad_dataset, batch_size=config.batch_size, shuffle=False)
        else:
            mhad_test_dataloader = DataLoader(
                mhad_dataset, batch_size=config.batch_size, shuffle=False)
            
    return mhad_train_dataloader, mhad_val_dataloader, mhad_test_dataloader

if __name__ == '__main__':
    mhad_train_dataloader, mhad_val_dataloader, mhad_test_dataloader = mhad_dataloader()
    