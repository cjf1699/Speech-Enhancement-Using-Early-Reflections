import torch
import torch.nn as nn
import torch.utils.data as data
import logging
import config as c
import matplotlib.pyplot as plt
import argparse
import warnings
import random
import numpy as np
import sys
from net import BeamEnhancer 
from DataPreprocessor import BeamSpecDataset, dataset_generator
from handler import *
sys.path.append('/home/cjf/workspace/201903_dereverLocEnhance/mir_eval_master/')
from mir_eval import separation as sep

# ================= read parameters from cmd, for run mode ====================

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,
                    default='test',
                    help='The name or function of the current task')
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

CUR_TASK = args.name
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


# ========================================================
# recording config
logging.basicConfig(level=logging.DEBUG,
                    filename='./log/' + CUR_TASK + '.log',
                    filemode='w',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger("train_logger")

# cmvn
spec_mean = np.load('abs_spec_mean.npy').astype(np.float32)
spec_std = np.load('abs_spec_std.npy').astype(np.float32)


# Train the model
def train_and_valid(learning_rate=c.lr, weight_decay=c.weight_decay, plot=True):
    """
    Train the model and run it on the valid set every epoch
    :param weight_decay: for L2 regularzition
    :param learning_rate: lr
    :param plot: draw the train/valid loss curve or not
    :return:
    """
    logger.info("This training task is set as following:\n \
                 num_epochs:{},\n \
                 learning_rate:{}\n \
                 dropout:{}\n \
                 weight_decay:{}\n \
                 =======================================".format(c.num_epochs, learning_rate, c.dropout, c.weight_decay))
    curr_lr = learning_rate

    # model define
    model = BeamEnhancer(input_size=c.input_size,
                         hidden_size=c.hidden_size,
                         output_size=c.output_size,
                         num_layers=c.num_layers,
                         dropout_v=c.dropout,
                         device=device).to(device)
    # print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # These parameters are for searching the best epoch to early stopping
    train_loss_curve, valid_loss_curve = [], []
    best_loss, avr_valid_loss = 1000000000000.0, 0.0

    best_epoch = 0
    best_model = None  # the best parameters

    for epoch in range(c.num_epochs):
        # 每一轮的 训练集/验证集 误差
        train_loss_per_epoch, valid_loss_per_epoch = 0.0, 0.0
        train_step_cnt, valid_step_cnt = 0, 0

        # 进入训练模式
        model.train()
        for current_dataset in dataset_generator(path=c.save_dir_for_SE, job_type='tr'):
            train_loader = data.DataLoader(dataset=current_dataset,
                                           batch_size=c.batch_size,
                                           shuffle=True)

            for step, (examples, labels) in enumerate(train_loader):
                # if step == 1:
                #     break
                train_step_cnt += 1
                # print(train_step_cnt)
                
                input_amplitude, input_phase = examples[0], examples[1] # The log amplitude spectrum of beam signal
                output_amplitude, output_phase = labels[0], labels[1] # The amplitude spectrum of clean signal
                # norm
                input_amplitude = (input_amplitude - torch.from_numpy(spec_mean)) / torch.from_numpy(spec_std + c.eps)

                input_feats_amp = input_amplitude.float().to(device)
                input_feats_pha = input_phase.float().to(device)
                output_feats_amp = output_amplitude.float().to(device)
                output_feats_pha = output_phase.float().to(device)
                mask = model(input_feats_amp)

                train_loss = criterion(mask * input_feats_amp, output_feats_amp * torch.cos(output_feats_pha - input_feats_pha))
                train_loss_per_epoch += train_loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                logger.info("Epoch [{}/{}], Step {}, train Loss: {:.4f}"
                            .format(epoch + 1, c.num_epochs, train_step_cnt, train_loss.item()))

        if plot:
            train_loss_curve.append(train_loss_per_epoch / train_step_cnt)

        if c.running_lr and epoch > 1 and (epoch + 1) % 2 == 0:
            curr_lr = curr_lr * (1 - c.decay)
            update_lr(optimizer, curr_lr)

        # valid every epoch
        # 进入验证模式

        model.eval()
        with torch.no_grad():
            for current_dataset in dataset_generator(path=c.save_dir_for_SE, job_type='cv'):
                valid_loader = data.DataLoader(dataset=current_dataset,
                                               batch_size=c.batch_size,
                                               shuffle=True)

                for step, (examples, labels) in enumerate(valid_loader):
                    # if step == 1:
                    #     break
                    valid_step_cnt += 1
                    # print(valid_step_cnt)
                    input_amplitude, input_phase = examples[0], examples[1]  # The log amplitude spectrum of beam signal
                    output_amplitude, output_phase = labels[0], labels[1]  # The amplitude spectrum of clean signal
                    # norm
                    input_amplitude = (input_amplitude - torch.from_numpy(spec_mean)) / torch.from_numpy(spec_std + c.eps)

                    input_feats_amp = input_amplitude.float().to(device)
                    input_feats_pha = input_phase.float().to(device)
                    output_feats_amp = output_amplitude.float().to(device)
                    output_feats_pha = output_phase.float().to(device)
                    mask = model(input_feats_amp)

                    valid_loss = criterion(mask * input_feats_amp, output_feats_amp * torch.cos(output_feats_pha - input_feats_pha))
                    valid_loss_per_epoch += valid_loss.item()

                    logger.info('The loss for the current batch:{}'.format(valid_loss))

            avr_valid_loss = valid_loss_per_epoch / valid_step_cnt

            logger.info('Epoch [{}/{}], the average loss on the valid set: {} '.format(epoch+1, c.num_epochs, avr_valid_loss))

            valid_loss_curve.append(avr_valid_loss)
            if avr_valid_loss < best_loss:
                best_loss = avr_valid_loss
                best_epoch, best_model = epoch, model.state_dict()

    # end for loop of epoch
    torch.save({
        'epoch': best_epoch,
        'state_dict': best_model,
        'loss': best_loss,
    }, './models/ckpoint_' + CUR_TASK + '.tar')

    logger.info('best epoch:{}, valid loss:{}'.format(best_epoch, best_loss))
    if plot:
        x = np.arange(c.num_epochs)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, train_loss_curve, 'b', label='Train Loss')
        ax.plot(x, valid_loss_curve, 'r', label='Valid Loss')
        plt.legend(loc='upper right')
        plt.savefig(CUR_TASK + '.jpg')
        plt.close()


# verify the model checkpoint
if __name__ == '__main__':
    train_and_valid()

