import config as c
import random
import time
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.io import wavfile as wav
import torch
import torch.utils.data as data
from scipy import signal
from hoa_params import get_encoder
from scipy.special import hankel2
from handler import *


class DataProcessor(object):
    """
    Convert the time-domain data (.wav) to HOADataset format:
    dict fo tensors:
    X: (C, T, F)
    Y: (360,)
    """

    def __init__(self, path, image_path, tf_list, snr_list, net='res', is_tr='tr',
                 is_speech=False, data_type='hoa'):

        # check pass
        self.anechoic_path = path
        self.image_path = image_path
        self.tf_list = tf_list  # tf of different rooms
        self.snr_list = snr_list  # various SNR
        self.net = net
        self.data_type = data_type  # hoa or stft
        self.is_tr = is_tr  # tr, cv, or te
        self.is_speech = is_speech
        print(self.is_speech)

        term = '/mnt/hd8t/cjf/random_reverb_wavs/full_freq/'
        if self.is_speech:
            term += 'speech/'
        if self.data_type == 'stft':
            term += 'STFT/'
        term += (self.is_tr + '/')
        self.save_path = term

        with open(self.anechoic_path, 'r') as f:
            _all_path = f.readlines()
        self.num_of_wavs = len(_all_path)

    def run(self):
        file_idx = 0

        for path in file_gen(self.anechoic_path):
            file_idx += 1
            # print(file_idx)

            print(self.data_type, self.is_tr, file_idx)
            content = path.strip().split(' ')
            adr, label = content[0], content[1]

            index = int(label) - 1
            random.shuffle(self.tf_list)
            TF_path = self.tf_list[0]

            if self.is_tr == 'tr':
                random.shuffle(self.snr_list)
                snr = self.snr_list[0]
            else:
                thres = self.num_of_wavs / len(self.snr_list)
                if file_idx < thres:
                    snr = self.snr_list[0]
                elif thres <= file_idx < 2 * thres:
                    snr = self.snr_list[1]
                elif 2 * thres <= file_idx < 3 * thres:
                    snr = self.snr_list[2]
                else:
                    snr = self.snr_list[3]

            dataset, cnt = transform(adr, TF_path, index, snr, self.data_type, self.is_speech)

            torch.save(dataset, self.save_path + 'DataSet_' + str(file_idx) + '.pt')


class HOADataSet(data.Dataset):
    """
    Generate the appropriate format DataSet.

    """

    def __init__(self, path, index, mean_name, std_name, max_name=None, norm=True):
        super(HOADataSet, self).__init__()
        self.readPath = path
        self.examples = torch.load(self.readPath + 'DataSet_' + str(index) + '.pt')
        self.X = self.examples['X']
        self.Y = self.examples['Y']
        if norm:    # do normalize
            if max_name == None:
                self.data_mean = torch.load(mean_name)
                self.data_std = torch.load(std_name)
                flag1 = (self.data_std == 0).any()
                if flag1:   raise RuntimeError('There exists 0 in std matrix!')
                self.X = (self.X - self.data_mean) / self.data_std
            else:
                self.data_max = torch.load(max_name)
                flag2 = (self.data_max == 0).any()
                if flag2:    warnings.warn('There exists all-zero dimension!')
                self.X = torch.from_numpy(np.where(self.X == 0, self.X, self.X / self.data_max))

    def __getitem__(self, index):
        _sample, _label = self.X[index], self.Y[index]

        return _sample, _label

    def __len__(self):
        return len(self.X)

    def __add__(self, other):
        self.X = torch.cat((self.X, other.X), dim=0)
        self.Y = torch.cat((self.Y, other.Y), dim=0)
        return self


class ERdataset(data.Dataset):
    def __init__(self, path, index, norm=None):
        super(ERdataset, self).__init__()
        self.readPath = path
        self.examples = torch.load(self.readPath + 'SEdata{}.pt'.format(index))
        self.X = self.examples['X']
        self.Y = self.examples['Y']
        if norm != None:
            self.data_mean = torch.load('SE_datamean.pt')
            self.data_std = torch.load('SE_datastd.pt')
            self.X = (self.X - self.data_mean) / self.data_std

    def __getitem__(self, index):
        _sample, _label = self.X[index], self.Y[index]
        return _sample, _label

    def __len__(self):
        return len(self.X)

    def __add__(self, other):
        self.X = torch.cat((self.X, other.X), dim=0)
        self.Y = torch.cat((self.Y, other.Y), dim=0)
        return self


class BeamSpecGenerator(object):
    """
    To generate STFT spectrum of beams which are obtained through beamforming applied to multi-array signals
    """
    def __init__(self, job_type, config_file=c, noisy=False):
        self.config = config_file
        self.job_type = job_type
        self.noisy = noisy
        self.anechoic_path = self.config.flist_dir + 'anechoic_mono_speech_' + self.job_type + '.flist'
        self.logger = logging.getLogger("gen_recorder")
        self.set_logger()

    def set_logger(self):
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(filename='./log/gen_{}_data.log'.format(self.job_type), mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def gen_spectrum(self):
        # choose room (RIR) according to job type
        if self.job_type == 'tr':
            tf_list = self.config.TRAIN_TF_LIST
            if self.noisy:
                random.shuffle(self.config.snr_list)
                snr = self.config.snr_list[0]
        elif self.job_type == 'cv':
            tf_list = self.config.VALID_TF_LIST
        elif self.job_type == 'tt':
            tf_list = self.config.TEST_TF_LIST

        # get paths of all anechoic wavs
        with open(self.anechoic_path, 'r') as f:
            all_wavs = f.readlines()
            num_of_wavs = len(all_wavs)
       
        for file_idx, path in enumerate(all_wavs): 
            self.logger.info('job_type:{} file_idx:{}'.format(self.job_type, file_idx+1))
            path = path.replace('data', 'data1')
            content = path.strip().split(' ')
            adr, label = content[0], content[1]
            sample_rate, wav_data_temp = wav.read(adr)
            assert sample_rate == self.config.fs and len(wav_data_temp.shape) == 1

            index = int(label) - 1
            random.shuffle(tf_list)
            TF_path = tf_list[0]
            TF = load_mat(TF_path)
            array_data_noise_free = get_array_signal(wav_data_temp, TF[index][0])
            
            # If noisy is claimed to be True, then add Gaussian White Noise to the array signal
            if self.noisy:
                if self.job_type == 'cv' or self.job_type == 'tt':
                    thres = num_of_wavs / len(self.config.snr_list)
                    if file_idx < thres:
                        snr = self.config.snr_list[0]
                    elif thres <= file_idx < 2 * thres:
                        snr = self.config.snr_list[1]
                    elif 2 * thres <= file_idx < 3 * thres:
                        snr = self.config.snr_list[2]
                    else:
                        snr = self.config.snr_list[3]
                array_data = awgn(array_data_noise_free, snr)
            else:
                array_data = array_data_noise_free
            angles_dict = get_reflection_angles(TF_path) # The 1st element of each value is the DOA of direct sound
            angles = angles_dict[index * 5]
            # Beamforming according to reflections directions
            extracted_sig = extract_sig(array_data, angles)
            assert extracted_sig.shape[1] == self.config.num_beams

            super_signal_1 = wav_data_temp
            direct_TF = full2direct_TF(TF_path)
            super_signal_2 = np.mean(get_array_signal(wav_data_temp, direct_TF[index][0]), axis=1)

            freq_array1, time_array1, beam_spectrums_temp = signal.stft(
                extracted_sig, self.config.fs, nperseg=self.config.frame_size, noverlap=self.config.n_overlap, nfft=self.config.fft_point, axis=0, padded=True)
            
            freq_array2, time_array2, clean_spectrums_temp = signal.stft(
                super_signal_1, self.config.fs, nperseg=self.config.frame_size, noverlap=self.config.n_overlap, nfft=self.config.fft_point, axis=0, padded=True)

            freq_array3, time_array3, direct_spectrums_temp = signal.stft(
                super_signal_2, self.config.fs, nperseg=self.config.frame_size, noverlap=self.config.n_overlap, nfft=self.config.fft_point, axis=0, padded=True)

            beam_spectrums = beam_spectrums_temp.transpose([1, 2, 0])   # change to (chan_, time, freq)
            clean_spectrums = clean_spectrums_temp.transpose([1, 0])    # change to (time, freq)
            direct_spectrums = direct_spectrums_temp.transpose([1, 0])  # change to (time, freq)

            if file_idx == 0:
                self.check_data_format(beam_spectrums, clean_spectrums)
                self.check_data_format(beam_spectrums, direct_spectrums)
                     
            save_dir = self.config.save_dir_for_SE + self.job_type + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            savemat(save_dir + '{}_{}.mat'.format(self.job_type, file_idx+1), {'beam_spec':beam_spectrums, 'clean_spec':clean_spectrums, 'direct_spec':direct_spectrums})

    def check_data_format(self, beam_spec, clean_spec):
        beam_shape = beam_spec.shape
        clean_shape = clean_spec.shape
        assert beam_shape[-1] == clean_shape[-1] == self.config.n_freq
        assert beam_shape[0] == self.config.num_beams
        assert beam_shape[1] == clean_shape[0]


class BeamSpecDataset(data.Dataset):
    def __init__(self, path, job_type, idx, supervisor='direct'):
        super(BeamSpecDataset, self).__init__()
        data_dict = loadmat(path + job_type + '/{}_{}.mat'.format(job_type, idx))
        self.supervisor = supervisor
        if self.supervisor == 'direct':
            Y_temp = data_dict['direct_spec']
        elif self.supervisor == 'clean':
            Y_temp = data_dict['clean_spec']
        else:
            raise RuntimeError("Unrecognized supervise signal")
        X_temp = data_dict['beam_spec'] 
        assert len(X_temp) == c.num_beams
        self.X = X_temp
        self.Y = np.tile(Y_temp, (len(X_temp), 1, 1))

    def __getitem__(self, index):
        samples_amp, samples_pha, labels_amp, labels_pha = np.abs(self.X[index]), np.angle(self.X[index]), np.abs(self.Y[index]), np.angle(self.Y[index]) 
        return (samples_amp, samples_pha), (labels_amp, labels_pha)
    
    def __len__(self):
        return len(self.X)

    def __add__(self, other):   # Note: need to modify though will not encounter for now. 20191221
        self.X = np.concatenate((self.X, other.X), axis=0)
        self.Y = np.concatenate((self.Y, other.Y), axis=0)
        return self


def dataset_generator(path, job_type):
    data_dir = path + job_type
    for _, _, files in os.walk(data_dir):
        all_dataset = files
    idx = 1
    while idx <= len(all_dataset):
        next_dataset = BeamSpecDataset(path, job_type, idx)
        idx += 1
        yield next_dataset 


def gen_SE_data(job_type):
    actor = BeamSpecGenerator(job_type=job_type)
    actor.gen_spectrum()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ser', type=str,
                        default='sh',  # sk : shengke 1 hao, sh: shrc
                        help='which server to use')
    parser.add_argument('--job_type', type=str,
                        default='tr',  
                        help='tr or cv or tt')

    args = parser.parse_args()

    SERVER = args.ser
    DATA_TYPE = args.job_type

    gen_SE_data(DATA_TYPE)


