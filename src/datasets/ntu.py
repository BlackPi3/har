import torch
from torch.utils.data import Dataset
from utils import config
import matplotlib.pyplot as plt

ACTIONS = {
    'A010': 0, # two hands front clapping
    'A007': 1, # right arm throw
    'A096': 2, # ?
    'A063': 3, # ?
    'A065': 4, # ?
    'A006': 5 # only pick up with right hand
}


class NTUDataset(Dataset):
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

    def __init__(self, skel_list, label_list):
        """
        :skel_list: skeleton data: List of arrays. Each array has shape (3, 3, N)
        """
        self.window = int(config.mhad_window_sec *
                          config.mhad_accel_sampling_rate)
        self.stride = int(config.mhad_stride_sec *
                          config.mhad_accel_sampling_rate)

        self.skel_windows, self.label_windows = self._create_windows(skel_list, label_list)

    def __len__(self):
        return len(self.skel_windows)

    def __getitem__(self, idx):
        return self.skel_windows[idx], self.label_windows[idx]

    def _create_windows(self, skel_list, label_list):
        skel_windows = []
        label_windows = []
        for skel, label in zip(skel_list, label_list):  # shape (3, 3, N)
            frames = skel.shape[2]
            for start in range(0, frames - self.window + 1, self.stride):
                end = start + self.window
                s_w = skel[:, :, start:end]
                l_w = ACTIONS[label]
                
                s_w = torch.tensor(s_w, dtype=config.dtype)
                l_w = torch.tensor(l_w, dtype=torch.long)

                # test
                # self.plot_joint(s_w, 0, 0)

                skel_windows.append(s_w)
                label_windows.append(l_w)


        skel_windows = torch.stack(skel_windows)
        label_windows = torch.stack(label_windows)

        return skel_windows, label_windows
