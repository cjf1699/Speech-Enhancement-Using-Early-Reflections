import numpy as np
import config as c
import torch
import warnings
from scipy.io import wavfile as wav
from scipy.io import loadmat
from scipy import signal
from hoa_params import get_encoder, get_y2
from encode import HOAencode
from signalprocess import *


def peak_detection(data):
    # check pass
    assert len(data.shape) == 1

    N = len(data)
    peaks = []
    for i in range(N):
        if i - c.sigma < 0 or i + c.sigma > N:
            if i - c.sigma < 0:
                tmp = torch.cat((data[i - c.sigma:], data[:i + c.sigma]))
            else:
                tmp = torch.cat((data[i - c.sigma:], data[0:(i + c.sigma) % N + 1]))
        else:
            tmp = data[i - c.sigma:i + c.sigma]

        if data[i] == torch.max(tmp) and data[i] > c.thres:
            peaks.append(i)
        # if plot:
        #     plt.figure()
        #     plt.plot(data.cpu().detach().numpy())
        #     plt.savefig('normal' + time.asctime(time.localtime(time.time())) + '.jpg')
        #     plt.close()

    return peaks  # 返回峰值的角度


def cal_recall(pred, label):
    # check pass
    if len(pred) * len(label) == 0:
        raise RuntimeError('标签或预测的列表为空！\n')

    err = [5, 10, 15]

    recall = np.zeros(len(err))
    for idx, wucha in enumerate(err):
        cnt = 0
        for item in label:
            for yuce in pred:
                diff = np.abs(item - yuce)
                if diff > 180:
                    diff = 360 - diff
                if diff <= wucha:
                    cnt += 1
                    break
        recall[idx] = cnt / len(label)

    return recall


def cal_precision(pred, label):
    # check pass
    if len(pred) * len(label) == 0:
        raise RuntimeError('标签或预测的列表为空！\n')
    err = [5, 10, 15]
    prec = np.zeros(len(err))
    for idx, wucha in enumerate(err):
        cnt = 0
        for yuce in pred:
            for item in label:
                diff = np.abs(item - yuce)
                if diff > 180:
                    diff = 360 - diff
                if diff <= wucha:
                    cnt += 1
                    break
        prec[idx] = cnt / len(pred)

    return prec


def cal_delay(s1, s2):
    """
    s1:reference signal
    s2:signal to be precessed
    return: sample points s2 preceeds s1. if negtive, it means that s2 is slower than s1
    """
    assert len(s1.shape) == len(s2.shape) == 1  # assume both signal has only one channel
    _s1, _s2 = list(s1), list(s2)  # transform both signal to list for convenience
    # padding
    if len(_s1) < len(_s2):
        _s1 += ([0] * (len(_s2) - len(_s1)))
    elif len(_s2) < len(_s1):
        _s2 += ([0] * (len(_s1) - len(_s2)))
    _len = 2 * len(_s1) - 1
    corr = np.correlate(_s1, _s2, 'full')
    idx = np.argmax(np.abs(corr))
    maxlag = _len // 2
    lags = list(range(-maxlag, maxlag + 1, 1))
    assert len(lags) == _len
    return lags[idx]


def time_domain_shift(signal, shift_point):
    """
    Shift signal in time domain, shift_time can be real number.

    For fraction part, shift it in frequency domain.
    """
    assert len(signal.shape) == 1 or (len(signal.shape) == 2 and signal.shape[1] == 1)  # must be mono signal.

    point_integer = int(shift_point)
    point_fraction = shift_point - point_integer
    s_len = signal.shape[0]

    _signal = np.zeros(s_len + 2 * c.tf_max_len, dtype=float)
    _signal[c.tf_max_len + point_integer:c.tf_max_len + point_integer + s_len] = signal

    _signal_fft = np.fft.fft(_signal)
    fft_len = _signal_fft.size
    half_len = (s_len + 1) // 2

    _signal_fft[:half_len] *= np.exp(- 1j * 2 * np.pi * point_fraction / fft_len * np.asarray(range(half_len)))
    _signal_fft[-1:-half_len:-1] = np.conj(_signal_fft[1:half_len])

    _signal = np.fft.ifft(_signal_fft).real
    return _signal


def enhance(block, angles):
    """
    :param block: given a block of signal
    :param angles: directions to be enchanced
    angles的第一个是直达声对应的角度
    return: shifted signals according to the delay
    """
    # implement 1
    '''
    rec_signal = extract_sig(block, angles)
    shifted_signals = np.zeros((rec_signal.shape[0] + 2 * c.tf_max_len, len(angles)), dtype=float)
    lags = np.zeros(len(angles))

    for id, angle in enumerate(angles):
        if id >= 1:
            lags[id] = cal_delay(rec_signal[:, 0], rec_signal[:, id]) # 这个值代表了该信号相对于第一个信号超前了多少个点

        shifted_signals[:, id] = time_domain_shift(rec_signal[:, id], lags[id])
    # fig, ax = plt.subplots(4, 1)
    # for i in range(4):  ax[i].plot(shifted_signals[:, i])
    # plt.savefig('each_channel.jpg')
    return shifted_signals
    '''
    # implement 2
    rec_signal = extract_sig(block, angles)
    sig_length = rec_signal.shape[0]
    aligned_signals = np.zeros(rec_signal.shape)
    lags = np.zeros(len(angles), dtype=int)

    for idx, angle in enumerate(angles):
        if idx == 0:
            aligned_signals[:, idx] = rec_signal[:, idx]
        else:
            lags[idx] = cal_delay(rec_signal[:, 0], rec_signal[:, idx])  # 这个值代表了该信号相对于第一个信号超前了多少个点
            # lags[idx] = lags[idx].astype(int)
            # print(lags[idx])
            if lags[idx] > 0:
                warnings.warn('It seems the direct signal is not the earliest')
                aligned_signals[lags[idx]:, idx] = rec_signal[0:sig_length - lags[idx], idx]
            else:
                aligned_signals[0:sig_length + lags[idx], idx] = rec_signal[-lags[idx]:, idx]
    return aligned_signals


def load_mat(path):
    ori_data = loadmat(path)

    keys = list(ori_data.keys())
    for key in keys:
        if key[0] != '_':
            data = ori_data[key]
            break

    return data


"""
def extract_sig(block, angles):
    
    #block是麦克风阵列采集到的信号，本函数对其按照angles指定的角度做波束形成，输出波束信号
    
    assert block.shape[1] == c.n_chan
    print(block.shape[0])
    s_fft = [] 
    for chan_idx in range(c.n_chan):
        s_fft.append(stft(block[:, chan_idx], size=c.frame_size, shift=c.frame_step, fading=True, ceil=True))
    s_fft_trans = np.array(s_fft).transpose([2, 0, 1])
    hoa = get_HOA(s_fft_trans)
    tt = []
    for t_idx in range(c.n_chan):
        tt.append(istft(s_fft_trans[:, t_idx, :].T, size=c.frame_size, shift=c.frame_step, fading=True))
    tt_trans = np.array(tt).T
    print(tt_trans.shape[0])

    
    t_hoa = []
    for hoa_idx in range(c.hoa_num):
        t_hoa.append(istft(hoa[:, hoa_idx, :].T, size=c.frame_size, shift=c.frame_step, fading=True))
    t_hoa_trans = np.array(t_hoa).T
    assert t_hoa_trans.shape[1] == c.hoa_num
    print(t_hoa_trans.shape[0])

    rec_signal = np.zeros((t_hoa_trans.shape[0], len(angles)), dtype=float) # each column is an enhanced signal

    Y = get_y2((180 - np.array(angles)) / 180 * np.pi, 0)   # 这里要用180度减，因为球谐函数定义的问题 get spherical harmonics
    #Y = get_y2(np.array(angles) / 180 * np.pi, 0)   # 若使用乔越的Y矩阵，则不需要用180度减 get spherical harmonics
    for az_idx, angle in enumerate(angles):
        rec_signal[:, az_idx] = t_hoa_trans.dot(Y[:, az_idx])     # beamforming
    return rec_signal
"""


def padding(directon, a, b):
    """

    :param directon: 'forward' or 'backward' , where to pad, front or back
    :param a: signal 1
    :param b: signal 2
    :return: the padded signal
    """
    len_a, len_b = a.shape[0], b.shape[0]
    diff = abs(len_a - len_b)
    if directon == 'backward':
        if len_a < len_b:
            a = np.concatenate((a, np.zeros(diff)))
            return a
        else:
            b = np.concatenate((b, np.zeros(diff)))
            return b
    else:
        if len_a < len_b:
            a = np.concatenate((np.zeros(diff), a))
            return a
        else:
            b = np.concatenate((np.zeros(diff), b))
            return b


def extract_sig(block, angles):
    # block是麦克风阵列采集到的信号，本函数对其按照angles指定的角度做波束形成，输出波束信号

    # print(block.shape[0])
    freq_array, time_array, s_fft = signal.stft(
        block, c.fs, window='hann', nperseg=c.frame_size, noverlap=c.n_overlap, nfft=c.fft_point, axis=0, padded=True)
    s_fft = s_fft[c.valid_freq_index, :, :]

    hoa = get_HOA(s_fft)

    t_array, t_hoa = signal.istft(
        hoa, c.fs, nperseg=c.frame_size, noverlap=c.n_overlap, nfft=c.fft_point, time_axis=-1, freq_axis=0)

    t_hoa = t_hoa.T
    # print('t_hoa:', t_hoa.shape[0])

    # t_hoa, _fs = HOAencode(block, c.fs, 4)
    assert t_hoa.shape[1] == c.hoa_num
    if t_hoa.shape[0] >= block.shape[0]:
        t_hoa_trans = t_hoa[0:block.shape[0], :]
    else:
        t_hoa_trans = np.concatenate((t_hoa, np.zeros((block.shape[0] - t_hoa.shape[0], c.hoa_num))))
    assert t_hoa_trans.shape[0] == block.shape[0]
    rec_signal = np.zeros((t_hoa_trans.shape[0], len(angles)), dtype=np.float32)  # each column is a beam signal

    Y = get_y2((180 - np.array(angles)) / 180 * np.pi, 0)  # get spherical harmonics
    for az_idx, angle in enumerate(angles):
        rec_signal[:, az_idx] = t_hoa_trans.dot(Y[:, az_idx])  # beamforming
    return rec_signal.astype(np.float32)


def check_overflow(wav_data):
    # warning: the current version only supports int16 type
    times = 0
    while (wav_data > 32767).any() or (wav_data < -32768).any():
        times += 1
        wav_data_temp = wav_data / c.overflow_damp
        wav_data = wav_data_temp
    return wav_data, times


def save_wav_for_hear(wav_data, name, _dir='./'):
    wav_data_int = wav_data.astype(np.int16)
    wav.write(_dir + name, c.fs, wav_data_int)


def sig2frames(block):
    """
    assume block is a mono channel signal
    return: n*frames
    """
    assert len(block.shape) == 1

    frames = []
    start, end = 0, 0
    while start + c.frame_size <= len(block):
        end = start + c.frame_size
        frames.append(block[start:end])
        start += c.frame_step
    return np.array(frames)


def get_array_signal(mono_data, tf, conv_style='same'):
    '''
    mono_data: mono speech signal
    tf: the hrir for n_chans  shape: (tf_len, n_chan)
    n_chan: the number of mics.
    return: multichannel received signals stimulates by the mono_data
    '''
    assert (len(tf.shape) == 2 and tf.shape[1] == c.n_chan)
    if conv_style == 'full':
        L = len(mono_data) + tf.shape[0] - 1
    elif conv_style == 'same':
        L = max(len(mono_data), tf.shape[0])
    elif conv_style == 'valid':
        L = max(len(mono_data), tf.shape[0]) - min(len(mono_data), tf.shape[0]) + 1
    else:
        raise RuntimeError("Unrecognized convolve style!")

    result = np.zeros((L, c.n_chan))
    for i in range(c.n_chan):
        result[:, i] = np.convolve(mono_data, tf[:, i], mode=conv_style)

    return result.astype(np.float32)


def full2direct_TF(TF_path):
    new_path = TF_path.replace('TF_result_fs_{}'.format(c.fs), 'TF_result_direct_fs_{}/aligned/'.format(c.fs))
    return load_mat(new_path)


def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / x.size
    npower = xpower / snr
    return (np.random.randn(*x.shape) * np.sqrt(npower) + x).astype(np.float32)


def cal_SDR_improve(clean, direct, enhance):
    """
    calculate a SDR1: direct to clean
    calculate a SDR2: enhance to clean
    return :SDR2 - SDR1 (improvement)
    """
    import sys
    sys.path.append('/home/cjf/workspace/201903_dereverLocEnhance/mir_eval_master/')
    from mir_eval import separation as sep

    SDR1, SIR1, SAR1, perm1 = sep.bss_eval_sources(clean, enhance, False)
    SDR2, SIR2, SAR2, perm2 = sep.bss_eval_sources(clean, direct, False)
    return SDR2, SDR1, SDR1 - SDR2


def cal_cmvn(path, num, prefix, name, ignore_axis=None):
    """
    given the data path, the total number of files and the prefix of data name, calculate the mean and the std,
    prefix: the prefix of data, not cmvn
    name: the name of cmvn result
    if ignore_axis is not None, then the cooresponding axis would be regarded as "amount dimension"（数量维度）
    """
    # assume the index of file name begins at 1
    if ignore_axis != None:
        total_cnt = 0
        for i in range(1, num + 1):
            print('calculating mean:', i)
            data = torch.load(path + 'tr/' + prefix + str(i) + '.pt')['X']
            total_cnt += data.shape[ignore_axis]
            if i == 1:
                sum_X_temp = torch.sum(data, dim=ignore_axis)
                std_X_temp = torch.sum(data ** 2, dim=ignore_axis)
            else:
                sum_X_temp += torch.sum(data, dim=ignore_axis)
                std_X_temp += torch.sum(data ** 2, dim=ignore_axis)
        X_mean = sum_X_temp / total_cnt
        X_std = np.sqrt(std_X_temp / total_cnt - X_mean ** 2)
        torch.save(X_mean, name + 'X_mean.pt')
        torch.save(X_std, name + 'X_std.pt')

    else:
        for i in range(1, num + 1):
            print('calculating mean:', i)
            data = torch.load(path + 'tr/' + prefix + str(i) + '.pt')['X']
            if i == 1:
                sum_X_temp = data
                std_X_temp = data ** 2
            else:
                sum_X_temp += data
                std_X_temp += data ** 2
        X_mean = sum_X_temp / num
        X_std = np.sqrt(std_X_temp / num - X_mean ** 2)
        torch.save(X_mean, name + 'X_mean.pt')
        torch.save(X_std, name + 'X_std.pt')

    if (X_std == 0).any():
        warnings.warn('There exists 0 in X_std, may occur ZeroDivisionError!')


# 20191128
def get_az(path):
    # check pass
    with open(path, 'rb') as f:
        records = np.fromfile(f)
        tmp = records.reshape(-1, 5)
        # 按照到达时间排序
        tmp = tmp[tmp[:, 0].argsort()]
        # aaa = tmp[0, :]
        index1 = np.where(tmp[:, 4] <= 1)[0]
        tmp1 = tmp[index1, :]
        index2 = np.where(tmp1[:, 3] == 0)[0]
        tmp2 = tmp1[index2]
        y = tmp2[:, 2]
        x = tmp2[:, 1]
        az = np.arctan2(y, x) / np.pi * 180
        # 小于0的角度，加上360
        az = np.where(az < 0, az + 360, az)
        az = np.round(az).astype(int)
        # 把角度归到距离最近的5的倍数
        for idx, angle in enumerate(az):
            if angle % 5 == 0:
                continue
            if angle % 5 <= 2:
                az[idx] = angle - angle % 5
            else:
                az[idx] = (angle + 5) - angle % 5
    return az


def get_reflection_angles(TF_path):
    """
    TF_path: represents a distict room
    return: a dict, whose keys are 0~355°, and values are the DOAs of 1-order reflections by 4 walls
    """
    dir2ref_dict = {}
    aa = TF_path.split('_')

    try:
        for i in range(1, 73):
            read_path = (c.IMAGE_PATH + 'RT60_' + aa[4] + '/dist_' + aa[6] +
                         '/source_{}.binary').format(i)
            assert os.path.exists(read_path)
            direct_az = (i - 1) * c.resolution
            reflection_azs = get_az(read_path)
            dir2ref_dict[direct_az] = reflection_azs
        return dir2ref_dict
    except:
        raise RuntimeError("No such directory or file: {}".format(read_path))


def label_gen(label, tf_path):
    # direc_id = int(label) // 5
    direc_id = int(label) - 1

    aa = tf_path.split('_')

    read_path = (c.IMAGE_PATH + 'RT60_' + aa[4] + '/dist_' + aa[6] +
                 '/source_{}.binary').format(direc_id + 1)
    source_az = get_az(read_path)
    # print(source_az)
    all_az = np.linspace(0, 359, 360)[:, np.newaxis]
    gaussian_hot = None
    for idx, az in enumerate(source_az):
        if idx == 0:
            gaussian_hot = np.exp(-((all_az - az) ** 2) / (c.std ** 2))
        else:
            gaussian_hot = np.hstack((gaussian_hot, np.exp(-((all_az - az) ** 2) / (c.std ** 2))))
    gaussian_hot = np.max(gaussian_hot, axis=1)
    # plt.figure()
    # plt.plot(gaussian_hot)
    # plt.show()
    # plt.close()
    return gaussian_hot


def get_HOA(s_fft):
    encoder = get_encoder()
    nFrames = s_fft.shape[2]
    hoa = np.zeros((c.n_freq, c.hoa_num, nFrames), dtype=np.complex64)

    for freq_index, freq in enumerate(c.valid_freq_array):
        hoa[freq_index, :, :] = encoder[freq_index, :, :].dot(s_fft[freq_index, :, :])  # 文献中的b （26）
    # flag3 = (hoa == 0).any()
    res = hoa.astype(np.complex64)
    return res


def transform(wav_path, tf_path, index, snr, data_type='stft', cut=True):
    """
    Used during being-applied stage. This method transforms a wavform in time-domain into HOA or STFT format
    :param wav_path:
    :param index: index corresponding to direct sound direction
    :param tf_path: a random-picked RIR
    :param snr: a random-picked snr
    :param data_type: HOA or STFT
    :return: data in HOA or STFT format, the multi-channel audio data and the number of examples tranfromed from this wav_path

    """
    sample_rate, wav_mono = wav.read(wav_path)
    assert c.fs == sample_rate
    if cut:
        wav_mono = clippout_silence(wav_mono)
        print('cut后的长度：', wav_mono.shape)

    TF = load_mat(tf_path)
    wav_data = get_array_signal(wav_mono, TF[index][0]).astype(np.float32)
    wav_data = awgn(wav_data, snr)

    dataset = {}
    dataset['X'] = []
    _, _, s_fft = signal.stft(
        wav_data, c.fs, nperseg=c.frame_size, noverlap=c.n_overlap, nfft=c.fft_point, axis=0, padded=False)
    s_fft = s_fft[c.valid_freq_index, :, :]

    # flag1 = (s_fft == 0).any()
    if data_type == 'hoa':
        s_fft = get_HOA(s_fft)
        # flag2 = (s_fft == 0).any()
    start, cnt = 0, 0

    while start + c.frames_per_block + 2 <= s_fft.shape[2]:
        # print(cnt)
        cnt += 1  #
        temp1 = torch.from_numpy(s_fft[:, :, start:start + c.frames_per_block + 2].real)
        temp2 = torch.from_numpy(s_fft[:, :, start:start + c.frames_per_block + 2].imag)
        temp = torch.cat((temp1, temp2), dim=1).permute(
            [1, 2, 0])  # change (freq, chan, time) to (chan, time, freq)
        if start == 0:
            dataset['X'] = temp
            input_dim = list(temp.shape)
        else:
            dataset['X'] = torch.cat((dataset['X'], temp), dim=0)
        start += (c.frames_per_block + 2)

    input_dim.insert(0, cnt)

    dataset['X'] = dataset['X'].reshape(input_dim)
    dataset['Y'] = torch.from_numpy(label_gen(str(index + 1), tf_path)).repeat(cnt, 1)
    return dataset, cnt


def file_gen(file_name):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            yield line


def clippout_silence(data):
    # ======= assert the data to be a mono signal ========
    assert len(data.shape) == 1
    silence_ids = np.where(np.abs(data) < c.gamma * np.mean(np.abs(data)))
    # print(silence_ids)
    new_data = np.delete(data, silence_ids)
    # check pass
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.plot(data)
    # ax2 = fig.add_subplot(212)
    # ax2.plot(new_data)
    # # print(len(data), len(new_data))
    # plt.show()
    return new_data


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # test get_az
    path = '/home/cjf/workspace/Matlab/RirsOfRooms/RT60_0.52589/dist_1.6324/source_17.binary'
    ref_az = get_az(path)
    # a = np.array([1, 3, 2, 9, 0, 10])
    # b = np.array([0, 0, 0, 0, 1, 3, 2, 9, 0, 10])
