import torch
import torch.nn as nn
import torch.utils.data as data
import config as c
import numpy as np
import logging
import argparse
from scipy import signal
from scipy.io import wavfile as wav
from net import BeamEnhancer
from DataPreprocessor import dataset_generator
from handler import cal_SDR_improve

# ================= read parameters from cmd, for run mode ====================

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,
                    default=None,
                    help='The name or function of the model being evaluated')
parser.add_argument('--ser', type=str,
                    default='sh',  # sk : shengke 1 hao, sh: shrc
                    help='use which server to run')
parser.add_argument('--debug', type=int,
                    default=0,
                    help='debug mode or not')
parser.add_argument('--gpu', type=int,
                    default=1,  # sk : shengke 1 hao, sh: shrc
                    help='use GPU or CPU')

args = parser.parse_args()

MODEL_NAME = args.name
SERVER = args.ser
DEVICE_TYPE = args.gpu
DEBUG = args.debug
# some directory and  Device configuration
if SERVER == 'sk':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if DEVICE_TYPE == 1 else torch.device('cpu')
elif SERVER == 'sh':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') if DEVICE_TYPE == 1 else torch.device('cpu')
elif SERVER == 'ship':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if DEVICE_TYPE == 1 else torch.device('cpu')
else:
    raise RuntimeError('Unrecognized server!')
model_path = './models/ckpoint_{}.tar'.format(MODEL_NAME)
checkpoint = torch.load(model_path)
model = BeamEnhancer(input_size=c.input_size, hidden_size=c.hidden_size, output_size=c.output_size,
                     num_layers=c.num_layers, dropout_v=c.dropout, device=device).to(device)
model.load_state_dict(checkpoint['state_dict'])
criterion = nn.MSELoss()
model.eval()

test_file_paths = open(c.flist_dir + 'anechoic_mono_speech_tt.flist', 'r')
all_test_files = test_file_paths.readlines()

logger = logging.getLogger("eval_logger")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='./log/eval_{}.log'.format(MODEL_NAME), mode='w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

total_test_loss = 0.0
total_SDR_improve = 0.0
wav_idx = 0
# cmvn
spec_mean = np.load('abs_spec_mean.npy').astype(np.float32)
spec_std = np.load('abs_spec_std.npy').astype(np.float32)
model.eval()

if __name__ == "__main__":
    with torch.no_grad():
        for current_dataset in dataset_generator(path=c.save_dir_for_SE, job_type='tt'):
            wav_path = all_test_files[wav_idx].strip().split(' ')[0].replace('data', 'data1')
            sample_rate, clean_wav = wav.read(wav_path)
            test_loader = data.DataLoader(dataset=current_dataset,
                                        batch_size=c.batch_size,
                                        shuffle=True)

            for step, (examples, labels) in enumerate(test_loader):
                input_amplitude, input_phase = examples[0], examples[1]  # The amplitude spectrum of beam signal
                output_amplitude, output_phase = labels[0], labels[1]  # The amplitude spectrum of clean signal
                # norm
                input_amplitude = (input_amplitude - torch.from_numpy(spec_mean)) / torch.from_numpy(spec_std + c.eps)

                input_feats_amp = input_amplitude.float().to(device)
                input_feats_pha = input_phase.float().to(device)
                output_feats_amp = output_amplitude.float().to(device)
                output_feats_pha = output_phase.float().to(device)
                mask = model(input_feats_amp)

                test_loss = criterion(mask * input_feats_amp, output_feats_amp * torch.cos(output_feats_pha - input_feats_pha))
                total_test_loss += test_loss.item()

                logger.info('The loss for the current batch:{}'.format(test_loss))
                rec_spec_amp = (input_feats_amp * mask).cpu().detach().numpy()
                rec_spec_pha = input_feats_pha.cpu().detach().numpy()

                rec_spec_temp = rec_spec_amp * np.exp(1j * rec_spec_pha)
                rec_spec = rec_spec_temp.transpose(2, 0, 1)
                t_array, rec_sig = signal.istft(
                        rec_spec, c.fs, nperseg=c.frame_size, noverlap=c.n_overlap, nfft=c.fft_point, time_axis=-1, freq_axis=0)

                rec_sig = rec_sig.T
                enh_sig_temp = np.mean(rec_sig, axis=1)
                try:
                    direct_sig = rec_sig[0:len(clean_wav), 0]
                    enh_sig = enh_sig_temp[0:len(clean_wav)]
                    SDR_bef_enh, SDR_aft_enh, SDR_imp = cal_SDR_improve(clean_wav, direct_sig, enh_sig)
                    total_SDR_improve += SDR_imp
                    if DEBUG and wav_idx < 10:
                        wav.write("{}_clean.wav".format(wav_idx+1), c.fs, (clean_wav/10).astype(np.int16))
                        wav.write("{}_direct.wav".format(wav_idx+1), c.fs, (direct_sig/10).astype(np.int16))
                        wav.write("{}_enh_{}dB.wav".format(wav_idx+1, SDR_imp.item()), c.fs, (enh_sig/10).astype(np.int16))
                    logger.info('SDR before enhancement:{}'.format(SDR_bef_enh))
                    logger.info('SDR after enhancement:{}'.format(SDR_aft_enh))
                    logger.info('The SDR improvement for this wav:{}'.format(SDR_imp.item()))
                except:
                    print("clean_wav shape:", clean_wav.shape)
                    print("enh_sig shape:", enh_sig.shape)
                    raise RuntimeError("The length of SDR candidates Must be equal.")
            wav_idx += 1

        avr_test_loss = total_test_loss / len(all_test_files)
        avr_SDR_improve = total_SDR_improve / len(all_test_files)
        logger.info('The SDR improvement for test set:{}'.format(avr_SDR_improve))

