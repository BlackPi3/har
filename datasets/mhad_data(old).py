import os
import numpy as np
import config
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
import random

num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE")) if config.cluster else 0

ACTIONS = {
    'a1': 0,
    'a2': 1,
    'a3': 2,
    'a4': 3,
    'a5': 4,
    'a6': 5,
    'a7': 6,
    'a8': 7,
    'a9': 8,
    'a10': 9,
    'a11': 10,
    'a12': 11,
    'a13': 12,
    'a14': 13,
    'a15': 14,
    'a16': 15,
    'a17': 16,
    'a18': 17,
    'a19': 18,
    'a20': 19,
    'a21': 20
}


class MHAD(Dataset):
    def __init__(self, prefix, inertial, skeleton):
        self.inertial = inertial  # (3, N)
        self.skeleton = skeleton  # (3, joints, N)
        self.label = ACTIONS[prefix.split('_')[0]]

    def __len__(self):
        return self.inertial.shape[1] - config.mhad_window_length * config.mhad_sampling_rate

    def __getitem__(self, idx):
        inertial = self.inertial[:, idx:idx +
                                 config.mhad_window_length * config.mhad_sampling_rate]  # (3, W)
        inertial = torch.tensor(inertial, dtype=config.dtype)
        skeleton = self.skeleton[:, :, idx:idx +
                                 config.mhad_window_length * config.mhad_sampling_rate]  # (3, 3, W)
        skeleton = torch.tensor(skeleton, dtype=config.dtype)
        label = torch.tensor(self.label, dtype=torch.long)

        return skeleton, inertial, label


class RandomStridedSampler(Sampler):
    def __init__(self, dataset, stride_samples):
        self.dataset = dataset
        self.stride = stride_samples

    def __len__(self):
        return len(range(0, len(self.dataset), self.stride))

    def __iter__(self):
        return iter(torch.randperm(len(self.dataset))[: len(self)])


class SequentialStridedSampler(Sampler):
    def __init__(self, dataset, stride_samples):
        self.dataset = dataset
        self.stride = stride_samples

    def __len__(self):
        return len(range(0, len(self.dataset), self.stride))

    def __iter__(self):
        return iter(range(0, len(self.dataset), self.stride))


# Load MHAD data pairs. inertial and skeleton data
file_pairs = {}  # dictionary of dictionary: {prefix: {'inertial': None, 'skeleton': None}}
for s in config.MHAD_TRAIN_S_IDS + config.MHAD_VAL_S_IDS + config.MHAD_TEST_S_IDS:
    s_dir = os.path.join(config.mhad_data_dir, s)

    files = sorted(os.listdir(s_dir))
    # group modalities
    for file in files:
        # This will be either 'inertial_std.npy' or 'skeleton_upsampled_normal.npy'
        suffix = "_".join(file.split('_')[3:])

        if suffix == config.mhad_inertial_file or suffix == config.mhad_skeleton_file:
            # pattern is 'a_s_t' where a is action, s is subject, t is trial.
            prefix = "_".join(file.split('_')[:3])
            if prefix not in file_pairs:
                file_pairs[prefix] = {'inertial': None, 'skeleton': None}

            # load modality. skeleton or inertial
            # inertial: (3, N), skeleton: (3, joints, N)
            modality = np.load(os.path.join(s_dir, file))

            if suffix == config.mhad_inertial_file:
                modality_type = 'inertial'
            elif suffix == config.mhad_skeleton_file:
                # (3, M, 3) right shoulder, right elbow, right wrist
                modality = modality[:, [8, 9, 10], :]
                modality_type = 'skeleton'

            # skeleton or inertial
            file_pairs[prefix][modality_type] = modality

        else:  # ignore other files
            continue


# train, val, test = [], [], []
data = []
for prefix, pair in file_pairs.items():
    # contains_train = any(s in prefix for s in config.MHAD_TRAIN_S_IDS)
    # contains_val = any(s in prefix for s in config.MHAD_VAL_S_IDS)
    # contains_test = any(s in prefix for s in config.MHAD_TEST_S_IDS)

    dataset = MHAD(prefix, pair['inertial'], pair['skeleton'])
    data.append(dataset)
    # if contains_train:
    #     train.append(dataset)
    # elif contains_val:
    #     val.append(dataset)
    # elif contains_test:
    #     test.append(dataset)

random.seed(0)
random.shuffle(data)

# Calculate split indices
total_len = len(data)
train_end = int(total_len * 0.6)
val_end = train_end + int(total_len * 0.2)

# Split the data
train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]


train_datasets = ConcatDataset(train)
val_datasets = ConcatDataset(val)
test_datasets = ConcatDataset(test)

stride = int(config.mhad_window_stride * config.mhad_sampling_rate)  # samples
train_loader = DataLoader(
    train_datasets,
    batch_size=config.batch_size,
    sampler=RandomStridedSampler(
        train_datasets, stride),
    num_workers=num_cpus
)
val_loader = DataLoader(
    val_datasets,
    batch_size=config.batch_size,
    sampler=SequentialStridedSampler(
        val_datasets, stride),
    num_workers=num_cpus
)
test_loader = DataLoader(
    test_datasets,
    batch_size=config.batch_size,
    sampler=SequentialStridedSampler(
        test_datasets, stride),
    num_workers=num_cpus
)
