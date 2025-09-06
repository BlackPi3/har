import os
import bisect
import torch
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset


# Helper functions for loading data (copied from legacy utils for self-containment)
def load_modality(filepath):
    """
    Loads modality from filepath and returns numpy array, or None if no file is found.
    :param filepath: File path to MM-Fit modality.
    :return: MM-Fit modality (numpy array).
    """
    try:
        mod = np.load(filepath)
    except FileNotFoundError as e:
        mod = None
        print("{}. Returning None".format(e))
    return mod


def load_labels(filepath):
    """
    Loads and reads CSV MM-Fit CSV label file.
    :param filepath: File path to a MM-Fit CSV label file.
    :return: List of lists containing label data, (Start Frame, End Frame, Repetition Count, Activity) for each
    exercise set.
    """
    labels = []
    with open(filepath, "r") as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            labels.append([int(line[0]), int(line[1]), int(line[2]), line[3]])
    return labels

# >>> MMFit Dataset class <<< #


class MMFit(Dataset):
    def __init__(self, pose_file, acc_file, labels_file, sensor_window_length, cluster=False, dtype=torch.float32):
        """
        for now we're only using the left wrist joint and the accelerometer sensor
        :param sensor_window_length: the length of the window in samples
        :param frame_window_length: the length of the window in samples
        :param cluster: whether running on cluster (for data subset selection)
        :param dtype: torch data type
        """
        self.sensor_window_length = sensor_window_length

        self.pose = torch.as_tensor(
            load_modality(pose_file)[
                :, :, [0, 1, 16, 17, 18]], dtype=dtype
        )  # (3, N, (frame, timestamps, left shoulder, left elbow, left wrist))
        self.acc = torch.as_tensor(
            load_modality(acc_file), dtype=dtype
        )  # (N, (frame, timestamps, XYZ))

        if not cluster:
            select = 1000
            self.pose = self.pose[:, :select, :]
            self.acc = self.acc[:select, :]

        self.ACTIONS = {
            "squats": 0,
            "lunges": 1,
            "bicep_curls": 2,
            "situps": 3,
            "pushups": 4,
            "tricep_extensions": 5,
            "dumbbell_rows": 6,
            "jumping_jacks": 7,
            "dumbbell_shoulder_press": 8,
            "lateral_shoulder_raises": 9,
            "non_activity": 10,
        }

        # list of length 30: (Start Frame, End Frame, Repetition Count, Activity Class)
        self.labels = load_labels(labels_file)
        self.start_frames = [row[0] for row in self.labels]
        self.end_frames = [row[1] for row in self.labels]

    def __len__(self):
        return self.pose.shape[1] - self.sensor_window_length

    def __getitem__(self, i):
        """
        dynamic windowing
        """

        # (3, joints, W) only keep the value. NOTE the permute
        # IMPORTANT
        sample_pose = self.pose[:, i: i + self.sensor_window_length, 2:].permute(
            0, 2, 1
        )
        # (3, W) only keep the value. NOTE the permute
        sample_acc = self.acc[i: i + self.sensor_window_length, 2:].permute(
            1, 0
        )  # IMPORTANT

        # - Label
        window_start_frame = self.pose[0, i, 0]
        window_end_frame = self.pose[0, i + self.sensor_window_length, 0]
        tolerance = (
            0.5  # if 50% of window is in the activity we still asign the activity lable
        )
        mid_point = (
            int(((window_end_frame - window_start_frame) * tolerance))
            + window_start_frame
        )  # important point
        index = bisect.bisect_right(self.start_frames, mid_point)
        label = "non_activity"
        if 0 < index and mid_point <= self.end_frames[index - 1]:
            label = self.labels[index - 1][3]

        return sample_pose, sample_acc, self.ACTIONS[label]


class RandomStridedSampler(Sampler):
    def __init__(self, dataset, stride_samples):
        """
        |param dataset|
        |param stride|
        """
        self.dataset = dataset
        self.stride = stride_samples

    def __len__(self):
        return len(range(0, len(self.dataset), self.stride))

    def __iter__(self):
        return iter(torch.randperm(len(self.dataset))[: len(self)])


class SequentialStridedSampler(Sampler):
    def __init__(self, dataset, stride_samples):
        """
        |param dataset|
        |param stride|
        """
        self.dataset = dataset
        self.stride = stride_samples

    def __len__(self):
        return len(range(0, len(self.dataset), self.stride))

    def __iter__(self):
        return iter(range(0, len(self.dataset), self.stride))


def unfold(model, dataset, pred_acc, batch_size=128, window_stride=20, device='cpu', sensor_window_length=300):
    """
    first wrap the dataset with a dataloader but the smampler is sequential.
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialStridedSampler(dataset, window_stride),
    )

    written = torch.zeros(pred_acc.shape[1], dtype=torch.bool).to(device=device)  # Track if a value was written

    global_start = 0  # Initialize the global starting point
    model.eval()
    with torch.no_grad():
        for pose, _, _ in dataloader:
            pose = pose.to(device, non_blocking=True)
            batch_pred_acc = model(pose)  # (batch, 3, N)

            for j in range(batch_pred_acc.shape[0]):  # for each batch
                start = global_start + j * window_stride
                end = start + sensor_window_length
                # Find elements that haven't been written to yet
                mask = written[start:end] == 0
                pred_acc[:, start:end][:, mask] = batch_pred_acc[j, :, mask]
                written[start:end][mask] = True

            # Update the global starting point for the next batch
            global_start += batch_pred_acc.shape[0] * window_stride


def build_mmfit_datasets(cfg):
    """
    Factory function to build MMFit train/val/test datasets based on config.
    This function is called by src.data.get_dataloaders().
    
    Args:
        cfg: Configuration object with mmfit-specific attributes
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) as ConcatDataset objects
    """
    # Default configuration values (fallback if not provided in cfg)
    DEFAULT_TRAIN_SUBJECTS = ['w01', 'w02', 'w03', 'w04', 'w06', 'w07', 'w08', 'w16', 'w17', 'w18', 'w00', 'w05']
    DEFAULT_VAL_SUBJECTS = ['w14', 'w15', 'w19', 'w20']
    DEFAULT_TEST_SUBJECTS = ['w09', 'w10', 'w11', 'w13']
    
    TRAIN_SIM_SUBJECTS = ['w08', 'w16', 'w17', 'w18', 'w00', 'w05']
    
    # Extract configuration values with fallbacks
    data_dir = getattr(cfg, 'data_dir', '../datasets/mm-fit/')  # Updated fallback path
    train_ids = getattr(cfg, 'train_subjects', DEFAULT_TRAIN_SUBJECTS)
    val_ids = getattr(cfg, 'val_subjects', DEFAULT_VAL_SUBJECTS) 
    test_ids = getattr(cfg, 'test_subjects', DEFAULT_TEST_SUBJECTS)
    
    pose_file = getattr(cfg, 'pose_file', 'pose_3d_upsample_normal.npy')
    acc_file = getattr(cfg, 'acc_file', 'sw_l_acc_std.npy')
    labels_file = getattr(cfg, 'labels_file', 'labels.csv')
    sim_acc_file = getattr(cfg, 'sim_acc_file', 'sw_l_sim_acc.npy')
    
    sensor_window_length = getattr(cfg, 'sensor_window_length', 300)  # 3 seconds * 100Hz
    
    # Check for mode configuration - determine if we should use simulated data
    use_simulated_data = getattr(cfg, 'use_simulated_data', False)
    
    train, val, test = [], [], []
    
    for w_id in train_ids + val_ids + test_ids:
        id_dir = os.path.join(data_dir, w_id)
        pose_file_path = os.path.join(id_dir, f"{w_id}_{pose_file}")
        acc_file_path = os.path.join(id_dir, f"{w_id}_{acc_file}")
        labels_file_path = os.path.join(id_dir, f"{w_id}_{labels_file}")

        # Use simulated accelerometer data for specific subjects if mode is combined
        if use_simulated_data and w_id in TRAIN_SIM_SUBJECTS:
            acc_file_path = os.path.join(id_dir, f"{w_id}_{sim_acc_file}")

        dataset = MMFit(
            pose_file=pose_file_path,
            acc_file=acc_file_path,
            labels_file=labels_file_path,
            sensor_window_length=sensor_window_length,
            cluster=getattr(cfg, 'cluster', False),
            dtype=torch.float32,  # Always use float32, don't overcomplicate
        )

        if w_id in train_ids:
            train.append(dataset)
        elif w_id in val_ids:
            val.append(dataset)
        elif w_id in test_ids:
            test.append(dataset)

    # Return ConcatDatasets
    train_dataset = ConcatDataset(train)
    val_dataset = ConcatDataset(val)
    test_dataset = ConcatDataset(test)
    
    return train_dataset, val_dataset, test_dataset


# >>> Legacy global data loading (for backward compatibility) <<< #
# This section is disabled to avoid conflicts with modular loading
# Uncomment if you need backward compatibility with legacy code

# try:
#     # Import legacy config if available
#     import sys
#     import os
#     legacy_utils_path = os.path.join(os.path.dirname(__file__), '..', 'legacy')
#     if os.path.exists(legacy_utils_path):
#         sys.path.append(legacy_utils_path)
#         from utils import config
#         from utils import utils
        
#         if hasattr(config, 'TRAIN_W_IDS'):  # Only execute if legacy config is available
#             TRAIN_SIM_W_IDS = ['w08', 'w16', 'w17', 'w18', 'w00', 'w05']

#             data_dir = config.mmfit_data_dir
#             train, val, test = [], [], []
#             for w_id in config.TRAIN_W_IDS + config.VAL_W_IDS + config.TEST_W_IDS:
#                 id_dir = os.path.join(data_dir, w_id)
#                 pose_file = os.path.join(id_dir, w_id + "_" + config.pose_file)
#                 acc_file = os.path.join(id_dir, w_id + "_" + config.acc_file)
#                 labels_file = os.path.join(id_dir, w_id + "_" + config.labels_file)

#                 if utils.args.mode == config.Mode.COMB and w_id in TRAIN_SIM_W_IDS:  # IMPORTANT
#                     # Simulated Accelerometer
#                     acc_file = os.path.join(id_dir, w_id + "_" + config.sim_acc_file)

#                 dataset = MMFit(
#                     pose_file=pose_file,
#                     acc_file=acc_file,
#                     labels_file=labels_file,
#                     sensor_window_length=config.sensor_window_length,
#                     cluster=config.cluster,
#                     dtype=config.dtype,
#                 )

#                 if w_id in config.TRAIN_W_IDS:
#                     train.append(dataset)
#                 elif w_id in config.VAL_W_IDS:
#                     val.append(dataset)
#                 elif w_id in config.TEST_W_IDS:
#                     test.append(dataset)

#             train_datasets = ConcatDataset(train)
#             val_datasets = ConcatDataset(val)
#             test_datasets = ConcatDataset(test)

#             train_loader = DataLoader(
#                 train_datasets,
#                 batch_size=config.batch_size,
#                 sampler=RandomStridedSampler(train_datasets, config.window_stride),
#                 # num_workers=config.num_cpus,
#             )
#             val_loader = DataLoader(
#                 val_datasets,
#                 batch_size=config.batch_size,
#                 sampler=SequentialStridedSampler(val_datasets, config.window_stride),
#                 # num_workers=config.num_cpus,
#             )
#             test_loader = DataLoader(
#                 test_datasets,
#                 batch_size=config.batch_size,
#                 sampler=SequentialStridedSampler(test_datasets, config.window_stride),
#                 # num_workers=config.num_cpus,
#             )
# except ImportError:
#     # If legacy utils are not available, skip backward compatibility
#     pass
