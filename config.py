from const import *
import numpy as np

# signal processing config
fs = 8000
frame_time = 0.032
frame_shift = 0.008
frame_size = int(frame_time * fs)
frame_step = int(frame_shift * fs)
n_overlap = frame_size - frame_step
fft_point = 2 ** int(np.ceil(np.log2(frame_size)))
frames_per_block = 20
block_size = frame_step * (frames_per_block - 1) + frame_size

n_chan = 32
tf_len = 8192 / 2

# ===========================================================================
# computer weights for each freq
freq_array = np.linspace(0, fs/2, fft_point//2+1)
valid_freq_min = 0
valid_freq_max = fs/2
valid_freq_index = (freq_array >= valid_freq_min) & (freq_array <= valid_freq_max) # 用于增强时，包不包含端点频率，SDR的improvement差别挺大的
valid_freq_array = freq_array[valid_freq_index]
n_freq = valid_freq_array.size
weight_array = np.ones_like(valid_freq_index)
# ============================================================================
# for HOA
min_freq, max_freq = 0, fs // 2
hoa_order = 4
hoa_num = 25

# scan params ==============================
az_num, el_num = 361, 1   # for compatibility with original code of decoder
scan_num = 72             # really used in this module
resolution = int(360 / scan_num)
ref_el_index = 0  # this program mainly run on equator.
az_max, az_min = 2*np.pi, 0
el_max, el_min = 0, np.pi
az_array = np.linspace(az_min, az_max, scan_num)
el_array = np.linspace(el_min, el_max, el_num) if el_num > 1 else np.asarray([np.pi / 2])

# ============================
mic_position = mic_position_32mic
array_radius = array_radius_32mic

speed_of_sound = 344  # m/s

# for time domain shift
tf_max_len = int(fs * 0.1)


# snr
snr_list = [10, 5, 0, -5]
# for clipping out the silence in a speech
gamma = 0.1

# paths
IMAGE_PATH = '/home/cjf/workspace/Matlab/RirsOfRooms/'
TF_PATH = '/mnt/hd8t/cjf/TF_result_fs_8000/'
TRAIN_TF_LIST = [TF_PATH + i for i in [
            'RT60_0.583_dist_1.9572_TF_Matrix.mat',
            'RT60_0.47149_dist_1.6555_TF_Matrix.mat',
            'RT60_0.31077_dist_1.6948_TF_Matrix.mat']]
VALID_TF_LIST = [TF_PATH + i for i in [
            'RT60_0.3687_dist_1.6557_TF_Matrix.mat',
            'RT60_0.56688_dist_1.3804_TF_Matrix.mat']]
TEST_TF_LIST = [TF_PATH + i for i in [
            'RT60_0.32684_dist_1.3816_TF_Matrix.mat',
            'RT60_0.52589_dist_1.6324_TF_Matrix.mat']]


# for overflow
overflow_damp = 1.5
# for ERs-attached beamforming
num_beams = 5
save_dir_for_SE = "/mnt/hd8t/cjf/SpeechEnhancement/EarlyReflections/"
flist_dir = "/home/cjf/workspace/flist/"
ref_mic_idx = 0
# Nerual Network training config
num_epochs = 50
batch_size = 5
running_lr = False
dropout = 0.0
decay = 0.04  # lr decay
lr = 0.001  # + list(1.0 / np.random.randint(1000, 2000, size=2))
weight_decay = 0.000
input_size = n_freq
output_size = n_freq
hidden_size = 600
num_layers = 2
eps = 1e-20
