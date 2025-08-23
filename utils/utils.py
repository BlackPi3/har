import os
import numpy as np
import csv
import datetime
import torch
import matplotlib.pyplot as plt
import sys

from utils import config


# stuff for mmfit_data.py
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


def get_subset(data, start=0, end=None):
    """
    Returns a subset of modality data.
    :param data: Modality (numpy array).
    :param start: Start frame of subset.
    :param end: End frame of subset.
    :return: Subset of data (numpy array).
    """
    if data is None:
        return None

    # Pose data
    if len(data.shape) == 3:
        if end is None:
            end = data[0, -1, 0]
        return data[
            :, np.where(((data[0, :, 0]) >= start) & ((data[0, :, 0]) <= end))[0], :
        ]

    # Accelerometer, gyroscope, magnetometer and heart-rate data
    else:
        if end is None:
            end = data[-1, 0]
        return data[np.where((data[:, 0] >= start) & (data[:, 0] <= end)), :][0]


# >>> FINDINGS <<< #
def find_latest_model(pattern):
    """
    find latest model from train_out
    """
    sorted_dirs = sorted(os.listdir(config.train_out_dir), reverse=True)
    for d in sorted_dirs:
        dir = os.path.join(config.train_out_dir, d)
        sorted_files = sorted(os.listdir(dir), reverse=True)
        for f in sorted_files:
            if pattern in f:
                return os.path.join(dir, f)


def find_f_in(autoencoder, val_loader):
    autoencoder.eval()
    with torch.no_grad():
        for pose, acc, labels in val_loader:
            out = autoencoder(acc)
            return out.shape[1] * out.shape[2]


# >>> SAVINGS <<< #
def get_save_dir():
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%d.%m/")
    time = current_date_time.strftime("%H.%M")

    train_out_dir = config.train_out_dir
    date_dir = os.path.join(train_out_dir, day)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)

    return date_dir, time

def save_model(model_state_dict, file_name):
    date_dir, time = get_save_dir()
    file = os.path.join(date_dir, time + "_" + file_name + ".pth")
    torch.save(model_state_dict, file)

def save_plot(
    epochs, best_epoch, train_metric_history, val_metric_history, metric, file_name
):
    date_dir, time = get_save_dir()

    ep = range(1, epochs + 2)  # 1 to epoch+1
    plt.figure(figsize=(10, 6))
    plt.plot(ep, train_metric_history, label=f"Training {metric}")
    plt.plot(ep, val_metric_history, label=f"Validation {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title(file_name)
    plt.legend()

    best_train_value = train_metric_history[best_epoch]
    best_val_value = val_metric_history[best_epoch]
    plt.text(best_epoch + 1, best_train_value, f"{best_train_value:.2f}", ha="right")
    plt.text(best_epoch + 1, best_val_value, f"{best_val_value:.2f}", ha="left")

    file = os.path.join(date_dir, time + "_" + file_name + ".png")
    plt.savefig(file)
    plt.close()


def save_log(log, file_name):
    date_dir, time = get_save_dir()
    file = os.path.join(date_dir, time + "_" + file_name + ".txt")
    with open(file, "w") as f:
        f.write(log)

# >>> ARGUMENTS <<< #
# - ARG CLASS
class Args:
    def __init__(self):
        self.seed = 0
        self.mode = config.Mode.REAL

        self._parse_args()

    def _parse_args(self):
        args = sys.argv
        for i, arg in enumerate(args):
            if arg == "-s":
                try:
                    self.seed = int(args[i + 1])
                except (ValueError, IndexError):
                    raise ValueError("seed value missing or not an integer")
            elif arg == "-m":
                try:
                    self.mode = args[i + 1]
                    if self.mode == "r":
                        config.mode = config.Mode.REAL
                    elif self.mode == "s":
                        config.mode = config.Mode.SIM
                    elif self.mode == "rs" or self.mode == "sr":
                        config.mode = config.Mode.COMB
                    else:
                        raise ValueError
                    
                    self.mode = config.mode
                except (ValueError, IndexError):
                    raise ValueError("mode value missing")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

args = Args()
set_seed(seed=args.seed)