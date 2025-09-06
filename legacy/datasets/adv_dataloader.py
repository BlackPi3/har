import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import config
from torch.utils.data import DataLoader
from datasets.ntu_dataloader import ntu_dataloader
from datasets.mhad_dataloader import mhad_dataloader
from datasets.adv_dataset import AdvDataset


def adv_dataloader():
    mhad_train_dataloader, mhad_val_dataloader, _ = mhad_dataloader()
    ntu_train_dataloader, ntu_val_dataloader, _ = ntu_dataloader()

    # Train
    mhad_train_accel_windows = mhad_train_dataloader.dataset.accel_windows
    mhad_train_skel_windows = mhad_train_dataloader.dataset.skel_windows
    mhad_train_label_windows = mhad_train_dataloader.dataset.label_windows

    ntu_train_skel_windows = ntu_train_dataloader.dataset.skel_windows
    ntu_train_label_windows = ntu_train_dataloader.dataset.label_windows

    adv_train_dataset = AdvDataset(mhad_train_accel_windows, mhad_train_skel_windows,
                                   mhad_train_label_windows, ntu_train_skel_windows, ntu_train_label_windows)
    
    adv_train_dataloader = DataLoader(adv_train_dataset, batch_size=config.batch_size, shuffle=True)

    # Validation
    mhad_val_accel_windows = mhad_val_dataloader.dataset.accel_windows
    mhad_val_skel_windows = mhad_val_dataloader.dataset.skel_windows
    mhad_val_label_windows = mhad_val_dataloader.dataset.label_windows

    ntu_val_skel_windows = ntu_val_dataloader.dataset.skel_windows
    ntu_val_label_windows = ntu_val_dataloader.dataset.label_windows

    adv_val_dataset = AdvDataset(mhad_val_accel_windows, mhad_val_skel_windows,
                                 mhad_val_label_windows, ntu_val_skel_windows, ntu_val_label_windows)
    
    adv_val_dataloader = DataLoader(adv_val_dataset, batch_size=config.batch_size, shuffle=False)

    
    # Test
    adv_test_dataloader = None

    return adv_train_dataloader, adv_val_dataloader, adv_test_dataloader

if __name__ == '__main__':
    adv_train_dataloader, adv_val_dataloader, _ = adv_dataloader()
    for accel, skel, label in adv_val_dataloader:
        print(accel.shape, skel.shape, label.shape)
        break