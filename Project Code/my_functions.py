import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fftshift, fft
from scipy.signal import lfilter


def up_sample(symbols, Tsym, Tsamp, complex):
    """
    Returns a numpy array of the samples spaced by Ts.
    INPUTS:
    Tsym: Symbol time
    Tsamp: Sample time
    OUTPUTS
    us: upsampled symbols
    """
    os = int(Tsym / Tsamp)
    if complex:
        us = np.zeros(len(symbols)*os, dtype=np.complex128)
    else:
        us = np.zeros(len(symbols)*os)
    us[::os] = symbols
    return us


def rcos(r, Rb, n, fs):
    """
    Inputs:
    r: Roll off factor
    Rb: Transmission rate 1/Tb
    n: half width pulse 
    fs: sample rate 1/Ts
    Outputs
    t : time vector of raised consine pulse
    rc: raised cosine pulse
    """
    t = np.arange(-n / Rb, n / Rb + 1 / fs, 1 / fs)
    rc = np.cos(np.pi * r * Rb * t) * np.sinc(Rb * t) / (1 - (2 * Rb * t * r) ** 2)
    return t, rc

def sig_pow(x):
    mag = abs(x)
    return np.sum(mag*mag) / len(mag)

def gen_rand_bits(N):
    """
    inputs:
    N: Number of bits for the frame 
    outputs:
    mess_bits: np.array of equiprobable 1s and 0s
    """
    return np.random.randint(2, size=N)

def polar_sym(mess_bits):
    """
    inputs:
    mess_bits: numpy array of 1s and 0s
    outputs:
    polar_sig: numpy array of +1s and -1s for each 1 and 0, respectively.
    """ 
    polar_sig = 2 * mess_bits - 1
    return polar_sig    

def thres_detect(signal):
    return (signal >= 0)

def BER_calc(sig1, sig2):
    """
    input:
    sig1: The message bits arrray set without error
    sig2: The message bits numpy array with error
    output:
    BER: The ratio of the number of bit errors by the total number of bits
    """
    err = sig1 != sig2
    return np.sum(err)/len(err)

def sig_eng(x):
    return np.sum(x*x)

def rrcos(r, Rsym, n, fsamp):
    """
    Inputs:
    r: Roll off factor
    Rsym: Transmission rate 1/Tb
    n: half width pulse 
    fs: sample rate 1/Ts
    Outputs
    t : time vector of raised consine pulse
    rrc: root raised cosine pulse
    """
    t = np.arange(-n / Rsym, n / Rsym + 1 / fsamp, 1 / fsamp)
    rrc = ((2 * r * (np.cos((1 + r) * np.pi * t * Rsym) + np.sin((1 - r) * np.pi * t * Rsym) / (4 * r * t * Rsym)))/
           (np.pi * np.sqrt(1 / Rsym) * (1 - (4 * r * t * Rsym)**2)))
    return t, rrc

def add_Gnoise(signal, SNRdB, bps=1, method='SNR'):
    """
    :param:
    signal: signal that will have noise added to it. Assumed Energy normalized if method == EbNo
    SNRdB: if method == 'SNR', SNR in dB, if method = 'EbNo', EbNo is dB
    bps: bits per symbol, used only with method = 'EbNo' for scaling
    method: method used to calculate sigma of the SNR. SNR uses signal power measurements, EbNo assumes noramlized energy
    :return: 
    sigma: standard deviation of the AWGN signal
    sigout: numpy array, signal plus noise, if signal is complex sigout is complex, if signal is real sigout it real
    """
    if method == 'SNR':
        snr = 10 ** (SNRdB / 10)
        sigma = np.sqrt(sig_pow(signal) / snr)
    elif method == 'EbNo':
        EbNo = (10 ** (SNRdB / 10)) * bps
        sigma = np.sqrt(1 / (2 * EbNo))
    if np.iscomplexobj(signal):
        sigwnoise = signal + np.random.normal(0, sigma, len(signal)) +1j*np.random.normal(0, sigma, len(signal))
    else:
        sigwnoise = signal + np.random.normal(0, sigma, len(signal))
    return sigma, sigwnoise

def qpsk_map(Ik):
    """
    Inputs:
    Ik: Message Bits assume even length array
    Outputs
    qpsk : complex signal 
    """
    qpsk = (1j-2j*Ik[0::2] + 1-2*Ik[1::2]) / np.sqrt(2)
    return qpsk

def qpsk_det(qpsk_syms):
    """
    :param:
    qpsk_syms: numpy array complex QPSK signal
    :return: 
    Ikhat: bit message estimate 
    """
    Ikhat = np.zeros(2*len(qpsk_syms))
    a = (np.imag(qpsk_syms) < 0)
    b = (np.real(qpsk_syms) < 0)
    Ikhat[0::2] = a
    Ikhat[1::2] = b
    Ikhat = Ikhat.astype('int32')
    return Ikhat
