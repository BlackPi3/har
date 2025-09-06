# region imports
import os
import torch
from enum import Enum, auto
import numpy as np
# endregion

# region device, dtype, ...
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    cluster = True  # IMPORTANT
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    cluster = False  # IMPORTANT

assert device.type != 'cpu', "only CPU available!"
dtype = torch.float32
np_dtype = np.float32

num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE")) if cluster else 1

config_dir = os.path.dirname(os.path.abspath(__file__))
train_out_dir = os.path.join(config_dir, "../../train_out")
# endregion

# region MMFit
mmfit_data_dir = '/netscratch/zolfaghari/data/mm-fit/' if cluster else os.path.join(
    config_dir, '../../data/mm-fit/')
EXPERIMENT_W_IDS = ['w01']
TRAIN_W_IDS = ['w01', 'w02', 'w03', 'w04', 'w06',
               'w07', 'w08', 'w16', 'w17', 'w18', 'w00', 'w05']
VAL_W_IDS = ['w14', 'w15', 'w19', 'w20']
# TEST_W_IDS = ['w00', 'w05', 'w12', 'w13', 'w20']
TEST_W_IDS = ['w09', 'w10', 'w11', 'w13']
sampling_rate = 100  # Hz
window_length = 3  # seconds
window_stride = int(0.2 * sampling_rate)
sensor_window_length = int(window_length * sampling_rate)
# endregion

# region UTD-MHAD
mhad_data_dir = '/netscratch/zolfaghari/data/UTD_MHAD/' if cluster else os.path.join(
    config_dir, '../../data/UTD_MHAD/')
MHAD_TRAIN_S_IDS = ['s1', 's3', 's5', 's7']
MHAD_VAL_S_IDS = ['s8', 's2']#, 's1', 's3']
MHAD_TEST_S_IDS = ['s4', 's6']
MHAD_SUBJECTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
mhad_actions_total = 21
mhad_actions_subset = ['a4', 'a5', 'a6', 'a7', 'a15', 'a21']
mhad_accel_sampling_rate = 50  # Hz
mhad_window_sec = 2  # 2 seconds
mhad_stride_sec = 0.1  # 200 ms
# endregion

# regsion NTU
ntu_data_dir = '/netscratch/zolfaghari/data/nturgbd/new_skel' if cluster else os.path.join(
    config_dir, '../../data/ntu/new_skel')
NTU_TRAIN_CAM_ID = ['C002', 'C003']
NTU_VAL_CAM_ID = ['C001']
NTU_ACTIONS_SUBSET = ['A006', 'A007', 'A010', 'A063', 'A065', 'A096']
ntu_fps = 30
# endregion
# ------------------------------------------------------------------------------------------------------------- #

# >>> MODEL <<< #
# - POSE2IMU Regressor
in_ch = 3
num_joints = 3
pose2imu_model_name = 'pose2imu'
best_pose2imu_seed = 1000000

# - Feature Extractor

# - Activity Classifier
ac_fin = 100
ac_num_classes = 11

# - Person Identifier
n_subjects = 21

# - PRETRAIN
pretrain_epochs = 15
pretrain_patience = 5

# - OPTIMIZATION
batch_size = 128
lr = 1e-4  # learning rate
epochs = 500
patience = 50  # patience for training early stop

# - SCENARIO 1
scenario1_name = 'scenario1'

# - SCENARIO 2
scenario2_alpha = 1  # Activity Coefficient
scenario2_beta = 100  # Similarity Coefficient
scenario2_name = 'scenario2'

# - SCENARIO 3
scenario3_alpha = 1  # Activity Coefficient
scenario3_beta = 1  # Person Coefficient
scenario3_gamma = 1  # Similarity Coefficient
scenario3_name = 'scenario3'

# - SCENARIO 4
scenario4_alpha = 10  # Activity Coefficient
scenario4_beta = 1  # Similarity Coefficient
scenario4_lambda = 1 # adv coefficient
scenario4_grad_pen = 1
scenario4_name = 'scenario4'


# # - AUTOENCODER
# ae_in_ch = 3
# ae_kernel_size = 7
# ae_kernel_stride = 2
# ae_dropout = 0.1
# ae_model_name = 'fe'

# # - CLASSIFIER
# fc_num_classes = 11
# fc_hidden_units = 100
# fc_dropout = 0.3
# fc_model_name = 'fc'

# ------------------------------------------------------------------------------------------------------------- #

# >>> PATHS <<< #
# -Files
original_pose_file = 'pose_3d.npy'
original_acc_file = 'sw_l_acc.npy'
pose_file = 'pose_3d_upsample_normal.npy'
acc_file = 'sw_l_acc_std.npy'
sim_acc_file = 'sw_l_sim_acc.npy'
labels_file = 'labels.csv'

mhad_inertial_file = 'inertial_std.npy'
mhad_skeleton_file = 'skeleton_upsample_normal.npy'

# region stuff from before


class Mode(Enum):
    REAL = auto()
    SIM = auto()
    COMB = auto()


class Dataset(Enum):
    MMFIT = auto()
    MHAD = auto()


mode = Mode.REAL  # IMPORTANT real is default
dataset = Dataset.MMFIT  # IMPORTANT select dataset
# endregion
