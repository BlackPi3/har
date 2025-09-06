import os
import bisect
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset

from utils import config
from utils import utils

num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE"))

# >>> MMFit Dataset class + WID <<< #
class MMFit(Dataset):
    def __init__(self, pose_file, acc_file, labels_file, w_id, sensor_window_length):
        """
        for now we're only using the left wrist joint and the accelerometer sensor
        :param sensor_window_length: the length of the window in samples
        :param frame_window_length: the length of the window in samples
        """
        self.sensor_window_length = sensor_window_length

        self.pose = torch.as_tensor(
            utils.load_modality(pose_file)[:, :, [0, 1, 16, 17, 18]], dtype=config.dtype
        )  # (3, N, (frame, timestamps, left shoulder, left elbow, left wrist))
        self.acc = torch.as_tensor(
            utils.load_modality(acc_file), dtype=config.dtype
        )  # (N, (frame, timestamps, XYZ))

        if not config.cluster:
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
        self.labels = utils.load_labels(labels_file)
        self.start_frames = [row[0] for row in self.labels]
        self.end_frames = [row[1] for row in self.labels]

        self.PARTICIAPNTS = {
            "w00": 0,
            "w01": 1,
            "w02": 2,
            "w03": 3,
            "w04": 4,
            'w05': 5,
            "w06": 6,
            "w07": 7,
            "w08": 8,
            'w09': 9,
            'w10': 10,
            'w11': 11,
            'w12': 12,
            'w13': 13,
            'w14': 14,
            'w15': 15,
            "w16": 16,
            "w17": 17,
            "w18": 18,
            "w19": 19,
            "w20": 20,
        }
        self.IDS = config.TRAIN_W_IDS
        self.w_id = w_id

    def __len__(self):
        return self.pose.shape[1] - self.sensor_window_length

    def __getitem__(self, i):
        """
        dynamic windowing
        """

        # (3, joints, W) only keep the value. NOTE the permute
        # IMPORTANT
        sample_pose = self.pose[:, i : i + self.sensor_window_length, 2:].permute(
            0, 2, 1
        )
        # (3, W) only keep the value. NOTE the permute
        sample_acc = self.acc[i : i + self.sensor_window_length, 2:].permute(
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

        return (
            sample_pose,
            sample_acc,
            self.ACTIONS[label],
            self.PARTICIAPNTS[self.w_id],
        )

# ------------------------------------------------------------------------------------------------------------- #

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


def unfold(model, dataset, pred_acc):
    """
    first wrap the dataset with a dataloader but the smampler is sequential.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=SequentialStridedSampler(dataset, config.window_stride),
    )

    written = torch.zeros(pred_acc.shape[1], dtype=torch.bool).to(
        device=config.device
    )  # Track if a value was written

    global_start = 0  # Initialize the global starting point
    model.eval()
    with torch.no_grad():
        for pose, _, _ in dataloader:
            batch_pred_acc = model(pose)  # (batch, 3, N)

            for j in range(batch_pred_acc.shape[0]):  # for each batch
                start = global_start + j * config.window_stride
                end = start + config.sensor_window_length
                # Find elements that haven't been written to yet
                mask = written[start:end] == 0
                pred_acc[:, start:end][:, mask] = batch_pred_acc[j, :, mask]
                written[start:end][mask] = True

            # Update the global starting point for the next batch
            global_start += batch_pred_acc.shape[0] * config.window_stride


# >>> load data, window, ... <<< #
data_dir = config.mmfit_data_dir
train, val, test = [], [], []
for w_id in config.TRAIN_W_IDS + config.VAL_W_IDS + config.TEST_W_IDS:
    id_dir = os.path.join(data_dir, w_id)
    pose_file = os.path.join(id_dir, w_id + "_" + config.pose_file)
    acc_file = os.path.join(id_dir, w_id + "_" + config.acc_file)
    labels_file = os.path.join(id_dir, w_id + "_" + config.labels_file)

    dataset = MMFit(
        pose_file=pose_file,
        acc_file=acc_file,
        labels_file=labels_file,
        w_id=w_id,
        sensor_window_length=config.sensor_window_length,
    )

    if w_id in config.TRAIN_W_IDS:
        train.append(dataset)
    elif w_id in config.VAL_W_IDS:
        val.append(dataset)
    elif w_id in config.TEST_W_IDS:
        test.append(dataset)

# ------------------------------------------------------------------------------------------------------------- #

train_datasets = ConcatDataset(train)
val_datasets = ConcatDataset(val)
test_datasets = ConcatDataset(test)

train_loader = DataLoader(
    train_datasets,
    batch_size=config.batch_size,
    sampler=RandomStridedSampler(train_datasets, config.window_stride),
    num_workers=num_cpus,
)
val_loader = DataLoader(
    val_datasets,
    batch_size=config.batch_size,
    sampler=SequentialStridedSampler(val_datasets, config.window_stride),
    num_workers=num_cpus,
)
test_loader = DataLoader(
    test_datasets,
    batch_size=config.batch_size,
    sampler=SequentialStridedSampler(test_datasets, config.window_stride),
    num_workers=num_cpus,
)
