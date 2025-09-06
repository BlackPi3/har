"""
Constants and configuration for the MMFit dataset.
"""

# Activity class mappings
ACTIONS = {
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

# Default subject splits
DEFAULT_TRAIN_SUBJECTS = [
    'w01', 'w02', 'w03', 'w04', 'w06', 'w07', 'w08', 
    'w16', 'w17', 'w18', 'w00', 'w05'
]

DEFAULT_VAL_SUBJECTS = ['w14', 'w15', 'w19', 'w20']

DEFAULT_TEST_SUBJECTS = ['w09', 'w10', 'w11', 'w13']

# Subjects that have simulated accelerometer data
TRAIN_SIM_SUBJECTS = ['w08', 'w16', 'w17', 'w18', 'w00', 'w05']

# Default file patterns
DEFAULT_POSE_FILE = 'pose_3d_upsample_normal.npy'
DEFAULT_ACC_FILE = 'sw_l_acc_std.npy'
DEFAULT_SIM_ACC_FILE = 'sw_l_sim_acc.npy'
DEFAULT_LABELS_FILE = 'labels.csv'

# Joint indices for MMFit pose data
# Indices: [frame, timestamps, left shoulder, left elbow, left wrist]
JOINT_INDICES = [0, 1, 16, 17, 18]
POSE_DATA_INDICES = slice(2, None)  # Skip frame and timestamp columns

# Default sensor parameters
DEFAULT_SENSOR_WINDOW_LENGTH = 300  # 3 seconds * 100Hz
DEFAULT_LABELING_TOLERANCE = 0.5  # 50% window overlap for activity labeling
