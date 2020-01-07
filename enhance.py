from handler import *
from scipy.io import wavfile as wav
import config as c
import numpy as np
import argparse
import logging
import time

parser = argparse.ArgumentParser()
parser.add_argument('--save_wav', type=int,
                    default=0,
                    help='1: save wav before and after enhancement 0: not save')
parser.add_argument('--save_plot', type=int,
                    default=0,
                    help='1: save pictures before and after enhancement 0: not save')
parser.add_argument('--name', type=str,
                    default='test',
                    help='the name of log file')

args = parser.parse_args()
hear = args.save_wav
plot = args.save_plot
log_name = args.name

logger = logging.getLogger("logger")
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('./log/record_{}.log'.format(log_name), mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
#logger.disabled = True


if __name__ == '__main__':
    tf_path = '/mnt/hd8t/cjf/TF_result_fs_8000/RT60_0.32684_dist_1.3816_TF_Matrix.mat'
    total_improvement = 0
    cnt = 0
    for line in file_gen('/home/cjf/workspace/flist/anechoic_mono_speech_tt.flist'):
        cnt += 1
        new_line = line.replace('data', 'data1')
        item = new_line.strip().split(' ')
        wav_path, label = item[0], item[1]
        
        sample_rate, s_mono = wav.read(wav_path)
        assert sample_rate == c.fs
        logger.info('单通道信号长度:{}'.format(s_mono.shape))
        SNR = 10
        index = int(label) - 1
        TF = load_mat(tf_path)
        s_multi = get_array_signal(s_mono, TF[index][0])
        # s_multi_noisy = awgn(s_multi, SNR)
        s_multi_noisy = s_multi   # 不加噪声的实验
        logger.info('多通道信号长度:{}'.format(s_multi_noisy.shape))
        angles = c.check_ref_angle[index * 5]
        
        logger.info('angles:{}'.format(angles))  
        
        logger.info('增强开始')
        shifted_signal = enhance(s_multi_noisy, angles)
        logger.info('增强信号的信号长度:{}'.format(shifted_signal.shape[0]))
        logger.info('增强结束')
        
        enhanced_signal = np.mean(shifted_signal, axis=1)
        ref_signal = shifted_signal[:, 0]
        ''' 
        delay_bet_ref_mono = cal_delay(s_mono, ref_signal)
        delay_bet_enhance_mono = cal_delay(s_mono, enhanced_signal)
        logger.info('直达声落后于单通道信号:{}'.format(-delay_bet_ref_mono))
        logger.info('增强信号落后于单通道信号{}:'.format(-delay_bet_enhance_mono))
        ref_signal1 = ref_signal[-delay_bet_ref_mono:-delay_bet_ref_mono+len(s_mono)]
        enhanced_signal1 = enhanced_signal[-delay_bet_enhance_mono:-delay_bet_enhance_mono+len(s_mono)]
        logger.info('ref shape:{}'.format(ref_signal1.shape))
        logger.info('enhance shape:{}'.format(enhanced_signal1.shape))
        '''
        ref_signal1 = ref_signal[0:len(s_mono)]
        enhanced_signal1 = enhanced_signal[0:len(s_mono)]

        logger.info('计算SDR开始')
        SDR0, SDR, improve = cal_SDR_improve(s_mono, ref_signal1, enhanced_signal1)
        logger.info('计算SDR结束')
        logger.info('反射声增强前:{}'.format(SDR0))
        logger.info('反射声增强后:{}'.format(SDR))
        logger.info('SDR improvement:{}'.format(improve))
        total_improvement += improve
        logger.info('=========================')
        if hear:
            logger.info('==================save wav=================')
            waiting_dict = {'s_mono':s_mono, 's_noisy':s_multi_noisy[:,0], 'direct_beam':ref_signal1, 'enhanced_beam_enh{:.2f}dB'.format(improve.item()):enhanced_signal1}
            result_dict = {}
            cnt_dict = {}
            for key in waiting_dict:
                new_sig, count = check_overflow(waiting_dict[key])
                result_dict[key] = new_sig
                cnt_dict[key] = count
            max_cnt = max(list(cnt_dict.values()))
            logger.info('最多放缩了{}次'.format(max_cnt))
            for key in result_dict:
                logger.info(key + ':' + '放缩了{}次'.format(cnt_dict[key]))
                if cnt_dict[key] < max_cnt: 
                    result_dict[key] = result_dict[key] / (c.overflow_damp ** (max_cnt - cnt_dict[key]))
                save_wav_for_hear(result_dict[key], '{}_'.format(cnt) + key + '.wav', './tmp/noNoise/')
            logger.info('=========================')
        if plot:
            logger.info('==================save picture=================')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1)

            ax[0].plot(s_mono)
            plt.title('clean/direct/enhance')
            ax[1].plot(ref_signal)
            ax[2].plot(enhanced_signal)

            plt.savefig('./pictures/noNoise/{}.jpg'.format(cnt))
            plt.close()
            '''
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(ref_signal1)
            ax[1].plot(enhanced_signal1)
            plt.title('direct/enhance after align')
            plt.savefig('./pictures/noNoise/{}_after_align.jpg'.format(cnt))
            plt.close()
            '''
            logger.info('=========================')
            
    mean_improvement = total_improvement / cnt
    logger.info('测试集上SDR平均提高:{}'.format(mean_improvement))
