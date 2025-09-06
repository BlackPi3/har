#region imports
import os
import sys
import numpy as np
from utils import config
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
#endregion

ACTIONS = {
    'a4': 0, # two hands front clapping
    'a5': 1, # right arm throw
    'a6': 2, # cross arms in chest
    'a7': 3, # basketball shooting
    'a15': 4, # tennis forehand swing
    'a21': 5 # pick up and throw
}

class MHADDataset(Dataset):
    # region test
    def plot_joint(self, skel, joint, label):
        plt.close()

        if isinstance(skel, torch.Tensor):
            skel = skel.cpu().numpy()

        fig, ax = plt.subplots(3, 1)
        fig.subplots_adjust(hspace=0.5)
        ax[0].plot(skel[0, joint, :], label="x")
        ax[0].legend()
        ax[1].plot(skel[1, joint, :], label="y")
        ax[1].legend()
        ax[2].plot(skel[2, joint, :], label="z")
        ax[2].legend()

        fig.suptitle(f'Joint: {joint}, Acitivty label: {label}')

        plt.show()
    # endregion

    def __init__(self, accel_list, skel_list, label_list):
        '''
        :accel_list: list of numpy arrays. each array is (3, N).
        :skel_list: list of numpy arrays. each array is (3, joints=3, N).
        :label_list: list of label corresponding to each item in accel_list.
        '''
        self.window = int(config.mhad_window_sec *
                          config.mhad_accel_sampling_rate)
        self.stride = int(config.mhad_stride_sec *
                          config.mhad_accel_sampling_rate)

        self.accel_windows, self.skel_windows, self.label_windows = self._create_windows(
            accel_list, skel_list, label_list)

    def __len__(self):
        return len(self.accel_windows)

    def __getitem__(self, idx):
        accel = self.accel_windows[idx]
        skel = self.skel_windows[idx]
        label = self.label_windows[idx]

        return accel, skel, label
    
    
    def _create_windows(self, accel_list, skel_list, label_list):
        accel_windows, skel_windows, label_windows = [], [], []

        for accel, skel, label in zip(accel_list, skel_list, label_list):

            assert accel.shape[1] == skel.shape[2], "Accel and Skel frames mismatch"
            frames = accel.shape[1]
            
            for start in range(0, frames - self.window + 1, self.stride):

                end = start + self.window

                a_w = accel[:, start:end]
                s_w = skel[:, :, start:end]
                l_w = ACTIONS[label]

                a_w = torch.tensor(a_w, dtype=config.dtype)
                s_w = torch.tensor(s_w, dtype=config.dtype)
                l_w = torch.tensor(l_w, dtype=torch.long)
                
                # test
                # self.plot_joint(s_w, 0, 0)

                accel_windows.append(a_w)
                skel_windows.append(s_w)
                label_windows.append(l_w)

        accel_windows = torch.stack(accel_windows)
        skel_windows = torch.stack(skel_windows)
        label_windows = torch.stack(label_windows)
        
        assert accel_windows.shape[0] == label_windows.shape[0] == skel_windows.shape[0], "Length of windows and label_windows are not equal"
        return accel_windows, skel_windows, label_windows
