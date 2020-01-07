# -*- coding: utf-8 -*-
"""
@author: yueqiao
Copyright by SHRC@PKU

HOA Encoding Process

encode microphone array recorded signal into HOA B-format signal with different order
将球形麦克风阵列录制的多通道信号编码为不同阶数的HOA B格式信号

目前支持的麦克风阵列类型：
“32mic” ： SHRC@PKU制作的32通道球形麦克风阵列
“eigenmike”：32通道4阶eigenmike录音球

"""

import math
import numpy as np
import scipy.signal
import scipy.io
from scipy import special 
from scipy.fftpack import fft, ifft
from scipy.io import savemat

def HOAencode(x_rec,fs,order,micType='32mic',coef=0.18,g1=1,g2=1,g3=1,g4=1):
    """
    32轨录音信号编码至HOA信号， 可选参数：阶数(order), 麦克风类型， 滤波系数(coef)
    arguments:
    x_rec: input microphone signal 麦克风录音信号，x*32 数组
    fs: sampling rate 采样率
    order: 编码阶数 （最高可取4）
    micType: 麦克风阵列类型（默认：SHRC球形阵列，可选eignmike）
    coef：正则化滤波系数
    g1,g2,g3,g4：增益补偿系数，默认为1

    return:
    x_HOA: 编码后的HOA信号数组 x*（N+1)^2，N为编码阶数
    fs：采样率

    """
    x = x_rec.copy()
    # 对信号进行增益补偿
    x[:,:8] *= g1
    x[:,8:16] *= g2
    x[:,16:24] *= g3
    x[:,24:32] *= g4
    channel = x[0,:].size
    ''' 
    w = scipy.signal.hamming(1025) #加窗
    w = w[:1024]
    wcopy = np.transpose(np.tile(w,(channel,1)))
    M = w.size              # window size
    N = 1024                # DFT size
    H = N//2                # Hop size
    hM1 = (M+1)//2
    hM2 = M//2
    '''
    
    w = scipy.signal.hanning(160) #加窗
    w = w[:160]
    wcopy = np.transpose(np.tile(w,(channel,1)))
    M = w.size              # window size
    N = 512                # DFT size
    H = M//2                # Hop size
    hM1 = (M+1)//2
    hM2 = M//2
    
    x = np.vstack((np.zeros((hM2,channel)),x)) #对信号前后两端进行补零
    x = np.vstack((x,np.zeros((hM1,channel))))
    pin = hM1
    pend = x[:,0].size - hM1
    Y = enmatrixY(micType,order)    #编码矩阵 
    E = np.linalg.pinv(Y)           #编码矩阵的伪逆矩阵
    f = np.linspace(0,fs/2,N//2+1)  #离散化的频率点（0~fs/2）
    Eq = matrixEQ(micType,order,f,coef=coef)   #均衡矩阵
    x_HOA = np.zeros((x[:,0].size,(order+1)**2))
    while pin<=pend:
    # --------analysis------- 频域进行编码操作
        x1 = x[pin-hM1:pin+hM2,:]
        x1 = np.multiply(x1,wcopy)
        X1 = np.transpose(fft(np.transpose(x1),N))
        X1_HOA = np.zeros((X1[:,0].size,(order+1)**2),dtype=complex)
        temp = np.dot(X1[:N//2+1,:],E)
        X1_HOA[:N//2+1,:] = np.multiply(temp,Eq)
        for i in range(1,N//2):
            X1_HOA[N//2+i,:] = np.conjugate(X1_HOA[N//2-i,:])
        X1_HOA[0,:] = X1_HOA[0,:].real
        X1_HOA[N//2,:] = X1_HOA[N//2,:].real
    # --------synthesis------- 时域进行HOA信号分帧重叠相加
        x1_HOA = (np.transpose(ifft(np.transpose(X1_HOA))))[0:M, :].real
        #print('x1_HOA:', x1_HOA.shape)
        #print('x_HOA:', x_HOA.shape)
        x_HOA[pin-hM1:pin+hM2,:] += x1_HOA
        pin += H
    x_HOA = np.delete(x_HOA,range(hM2),0)
    x_HOA = np.delete(x_HOA,range(x_HOA[:,0].size-hM1,x_HOA[:,0].size),0)
    return x_HOA,fs

def checkGain(x):
    """
    录音球信号的增益均衡检查。由于硬件问题，SHRC球阵录得的32通道信号每8通道为一组，有时会出现不同组之间信号幅值相差两倍的现象；
    为解决此问题，需要在编码前调整不同通道信号的增益使之幅值均衡。

    arguments:
    x: input signal 输入麦克风录制信号 x*32 数组

    return:
    gain1, gain2, gain3, gain4: 四个8通道分组的增益（取值为1或0.5）

    """
    gain1 = 1
    gain2 = 1
    gain3 = 1
    gain4 = 1
    xband1 = x[:,:8].flatten()
    xband2 = x[:,8:16].flatten()
    xband3 = x[:,16:24].flatten()
    xband4 = x[:,24:32].flatten()
    Eband1 = sum(np.multiply(xband1,xband1))
    Eband2 = sum(np.multiply(xband2,xband2))
    Eband3 = sum(np.multiply(xband3,xband3))
    Eband4 = sum(np.multiply(xband4,xband4))
    if Eband2 >= 3*Eband1: #检测不同分组的信号总能量是否接近，阈值为相差3倍
        gain2 = 0.5
    if Eband3 >= 3*Eband1:
        gain3 = 0.5
    if Eband4 >= 3*Eband1:
        gain4 = 0.5
    return gain1,gain2,gain3,gain4

def HOAencodePerFrame(x,fs,order,micType='32mic',coef=3e-2):
    """
    单帧信号的HOA编码
    与HOAencode功能类似，仅省去了overlap-add操作，适用于较短录音文件
    """
    N = x[:,0].size              # DFT size
    #w = w/sum(w)
    Y = enmatrixY(micType,order)
    E = np.linalg.pinv(Y)
    f = np.linspace(0,fs/2,N//2+1)
    Eq = matrixEQ(micType,order,f,coef=coef)
    x_HOA = np.zeros((x[:,0].size,(order+1)**2))
    # --------analysis-------
    X = np.transpose(fft(np.transpose(x),N))
    X_HOA = np.zeros((X[:,0].size,(order+1)**2),dtype=complex)
    temp = np.dot(X[:N//2+1],E)
    X_HOA[:N//2+1,:] = np.multiply(temp,Eq)
    for i in range(1,N//2):
        X_HOA[N//2+i,:] = np.conjugate(X_HOA[N//2-i,:])
        X_HOA[0,:] = X_HOA[0,:].real
        X_HOA[N//2,:] = X_HOA[N//2,:].real
       # X1_HOA[0,:]=X1_HOA[0,:].real
    # --------synthesis-------
    x_HOA = (np.transpose(ifft(np.transpose(X_HOA)))).real
    return x_HOA




def micposi(micType):
    """
    导入不同种类麦克风阵列中各麦克风的位置坐标
    格式：（elevation, azimuth) in radius, spherical coordinates
    具体方向规定请见技术报告

    argument:
    micType: 麦克风阵列类型

    return:
    mic_posi: 麦克风位置坐标数组 32*2

    """
    if micType == '32mic':
        mic_posi = np.array([ 
(2.82461506444509,	5.83490178427917),
(2.40252848419645,	4.71967503199637),
(1.98733490064365,	4.08383306385527),
(1.36527035801541,	4.28482758496428),
(0.758660117159106,	4.00107654045027),
(1.75478325409642,	4.81153169616892),
(2.11781160814468,	5.49663925097614),
(2.21400476100161,	0),
(1.72846444191660,	0.481275842819644),
(1.60073751619372,	6.04217170897984),
(1.47176503380847,	5.40184556297591),
(1.02903291190296,	4.89080929613175),
(0.914463746976567,	5.77422730719281),
(1.12682595776279,	0.222238377864230),
(0.497512595010939,	0.475772470233098),
(0.386235734178215,	5.08014156747802),
(2.64408005857885,	3.61736512382289),
(2.75535691941158,	1.93854891388823),
(2.38293253643069,	0.859483886860477),
(2.11255974168683,	1.74921664254196),
(2.22712890661323,	2.63263465360302),
(2.01476669582700,	3.36383103145402),
(1.41312821167320,	3.62286849640944),
(1.54085513739607,	2.90057905539004),
(1.66982761978132,	2.26025290938612),
(1.77632229557439,	1.14323493137448),
(1.38680939949338,	1.66993904257913),
(1.02378104544511,	2.35504659738635),
(0.927587892588179,	3.14159265358979),
(0.316977589144701,	2.69330913068937),
(0.739064169393344,	1.57808237840658),
(1.15425775294615,	0.942240410265476)])
        return mic_posi
    elif micType == 'eigenmike':
        mic_posi = np.array([
(1.20427718387609,	0),
(1.57079632679490,	0.558505360638186),
(1.93731546971371,	0),
(1.57079632679490,	5.72467994654140),
(0.558505360638186,	0),
(0.959931088596881,	0.785398163397448),
(1.57079632679490,	1.20427718387609),
(2.18166156499291,  0.785398163397448),
(2.58308729295161,	0),
(2.18166156499291,	5.49778714378214),
(1.57079632679490,	5.07890812330350),
(0.959931088596881,	5.49778714378214),
(0.366519142918809,	1.58824961931484),
(1.01229096615671,   1.57079632679490),
(2.11184839491314,	1.57079632679490),
(2.77507351067098,	1.55334303427495),
(1.20427718387609,	3.14159265358979),
(1.57079632679490,	3.70009801422798),
(1.93731546971371,	3.14159265358979),
(1.57079632679490,	2.58308729295161),
(0.558505360638186,	3.14159265358979),
(0.959931088596881,	3.92699081698724),
(1.57079632679490,	4.34586983746588),
(2.18166156499291,	3.92699081698724),
(2.58308729295161,	3.14159265358979),
(2.18166156499291,	2.35619449019235),
(1.57079632679490,	1.93731546971371),
(0.959931088596881,	2.35619449019235),
(0.366519142918809,	4.69493568786475),
(1.01229096615671,	4.71238898038469),
(2.12930168743308,	4.71238898038469),
(2.77507351067098,	4.72984227290463)])
        return mic_posi
    else:
        raise ValueError("wrong micType!")
        
def enmatrixY(micType,order):
    """
    导入编码矩阵Y(已提前算好)

    arguments:
    micType: 麦克风阵列类型
    order: 编码阶数

    return:
    Y: 编码矩阵
    """
    if micType == '32mic':
        #Y = scipy.io.loadmat("./matfile/my32mic-Y.mat")['data'].T  # modidied by cjf in 20191127
        Y = scipy.io.loadmat("./matfile/32mic-Y.mat")['Y']  # modidied by cjf in 20191127
        return Y[:(order+1)**2,:]
    elif micType =='eigenmike':
        Y = scipy.io.loadmat("./matfile/eigenmike-Y.mat")['Y']
        return Y[:(order+1)**2,:]
    else:
        raise ValueError("wrong micType!")


def matrixEQ(micType,order,f,coef):
    """
    计算均衡矩阵EQ

    arguments:
    micType: 麦克风阵列类型
    order: 编码阶数
    f: 离散频率数组
    coef: 正则化均衡系数

    return:
    EQ: 均衡矩阵 x*(N+1)^2, N为阶数
    """

    if micType =='32mic':
        r = 50e-3
    elif micType =='eigenmike':
        r = 42e-3
    else:
        raise ValueError("wrong micType!")
    veq = np.vectorize(eq)
    EQ = np.zeros((np.size(f),(order+1)**2),dtype=complex)
    for i in range(order+1):
        for j in range(2*i+1):
            EQ[:,i**2+j] = veq(f,r,i,coef)        
    return EQ


def eq(x,r,n,coef):
    """
    计算均衡矩阵中各元素数值

    argument:
    x: 某一单一频率值
    r: 球形阵列半径
    n: 阶数
    coef: 均衡系数

    return:
    out：均衡矩阵的一个元素
    """
    kr = 2*np.pi*x*r/344 + (x==0)*np.finfo(float).eps
    hnde = math.sqrt(np.pi/(2*kr))*(n/kr*special.hankel2(n+1/2,kr)-special.hankel2(n+3/2,kr))
    bn = (-1j)**(n+1)*(-1)**n/(kr**2*hnde)
#    bn = 1j/(kr**2*hnde)
    out = np.conjugate(bn)/((abs(bn))**2+coef**2)
    #out = (np.conjugate(1j/hnde)*(kr**2))/((coef**2)*(kr**4)-1/(hnde**2))/(-1j)**n
    return out
  
if __name__ == '__main__':
    savemat('mic_qiao.mat', {'mic': micposi('32mic')})
