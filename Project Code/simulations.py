import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.stats import norm

import my_functions
import hamming


def run_simulation(modulation='BPSK', code='7_4', N=2002000):
    """
    Inputs:
    modulation (str): 'BPSK' or 'QPSK', specifies modulation technique
    code (str): '7_4', '15_11', or '31_26', specifies which hamming code to use
    N (int): number of bits per simulation
    Outputs:
    SNR_dB (numpy.array()): SNR values used in simulation
    BER (numpy.array()): BER result for each value in SNR_dB
    
    This simulation runs with increasing SNR values, starting with -2, until the acheived BER is below 5*10^-5
    """
    np.random.seed(0)
    
    print('MODULATION: ' + modulation)
    print('CODE: HAMMING ' + code)
    print('NUMBER OF BITS: ' + str(N))
    
    # Generate Random Bits
    bits = my_functions.gen_rand_bits(N)
    
    # Generate root-raised cosine pulse
    r = 0.25
    Rb = 10
    n = 5
    os = 8
    fs = os * Rb

    t_rrc, rrc = my_functions.rrcos(r, Rb, n, fs)
    rrc = rrc / np.sqrt(my_functions.sig_eng(rrc))
    delay = int(len(t_rrc - 1) / 2)
    
    # Encode bits
    if (code == '7_4'):
        encoded = hamming.encoder_7_4(bits)
    elif (code == '15_11'):
        encoded = hamming.encoder_15_11(bits)
    elif (code == '31_26'):
        encoded = hamming.encoder_31_26(bits)
    else:
        raise ValueError('Invalid Encoding Scheme')
    
    # Modulate
    if (modulation == 'BPSK'):
        mod = my_functions.polar_sym(encoded)
    elif (modulation == 'QPSK'):
        mod = my_functions.qpsk_map(encoded)
    else:
        raise ValueError('Invalid Modulation Technique')
    
    # Up sample and pulse shape
    if (modulation == 'BPSK'):
        upsample = my_functions.up_sample(mod, 1 / Rb, 1 / fs, False)
    else:
        upsample = my_functions.up_sample(mod, 1 / Rb, 1 / fs, True)
    
    upsample = np.hstack((upsample,np.zeros(delay)))
    signal = lfilter(rrc, [1], upsample)
    signal = signal[delay:]
    
    # Creat BER array
    BER = np.array([])
    SNR_dB = np.array([])
    current_SNR = -2
    current_BER = 1
    
    # Simulate channel at each SNR
    while current_BER > 0.00005:
        np.random.seed(0)
        print('Simulating with SNR: ' + str(current_SNR))
        SNR_dB = np.append(SNR_dB, current_SNR) 
        # Add noise to sigal
        sigma, noisy_signal = my_functions.add_Gnoise(signal, current_SNR)
        # Send noisy signal through matched filter
        if (modulation == 'BPSK'):
            noisy_signal = np.hstack((noisy_signal,np.zeros(delay)))
        else:
            noisy_signal = np.hstack((noisy_signal,np.zeros(delay, dtype=np.complex128)))
        received_signal = lfilter(rrc, [1], noisy_signal)
        received_signal = received_signal[delay:]
        # Sample received signal
        samples = received_signal[::8]
        # Determine received bits
        if (modulation == 'BPSK'):
            received_bits = my_functions.thres_detect(samples)
        elif (modulation == 'QPSK'):
            received_bits = my_functions.qpsk_det(samples)
        # Decode bits
        if (code == '7_4'):
            decoded = hamming.decoder_7_4(received_bits)
        elif (code == '15_11'):
            decoded = hamming.decoder_15_11(received_bits)
        elif (code == '31_26'):
            decoded = hamming.decoder_31_26(received_bits)
        # Calculate bit error rate
        current_BER = my_functions.BER_calc(decoded, bits)
        print('BER ' + str(current_BER))
        BER = np.append(BER, current_BER)
        current_SNR += 0.25

    return SNR_dB, BER


def main():
    # BPSK Simulations and Plot
    SNR_BPSK_7_4, BER_BPSK_7_4 = run_simulation(modulation='BPSK', code='7_4')
    SNR_BPSK_15_11, BER_BPSK_15_11 = run_simulation(modulation='BPSK', code='15_11')
    SNR_BPSK_31_26, BER_BPSK_31_26 = run_simulation(modulation='BPSK', code='31_26')
    plt.figure()
    plt.plot(SNR_BPSK_7_4, BER_BPSK_7_4, 'r')
    plt.plot(SNR_BPSK_15_11, BER_BPSK_15_11, 'g')
    plt.plot(SNR_BPSK_31_26, BER_BPSK_31_26, 'b')
    plt.yscale('log')
    plt.ylabel('BER')
    plt.xlabel('SNR')
    plt.grid()
    # QPSK Simulations and Plot
    SNR_QPSK_7_4, BER_QPSK_7_4 = run_simulation(modulation='QPSK', code='7_4')
    SNR_QPSK_15_11, BER_QPSK_15_11 = run_simulation(modulation='QPSK', code='15_11')
    SNR_QPSK_31_26, BER_QPSK_31_26 = run_simulation(modulation='QPSK', code='31_26')
    plt.figure()
    plt.plot(SNR_QPSK_7_4, BER_QPSK_7_4, 'r')
    plt.plot(SNR_QPSK_15_11, BER_QPSK_15_11, 'g')
    plt.plot(SNR_QPSK_31_26, BER_QPSK_31_26, 'b')
    plt.yscale('log')
    plt.ylabel('BER')
    plt.xlabel('SNR')
    plt.grid()
    plt.show()
    
    


if __name__ == '__main__':
    main()
