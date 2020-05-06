import numpy as np

def encoder_7_4(Ik):
    """
    Inputs:
    Ik: Message Bits assume array length is integer divisble by 4
    Outputs
    Ikcws : Message bits in code words, first dimension is 1
    """
    G = np.array(([1, 0, 0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 1, 0, 1, 1]))
    Ik_new = Ik.copy()
    Ik_new = Ik_new.reshape(-1, 4)
    Ikcws = (Ik_new @ G) % 2
    Ikcws = Ikcws.flatten()
    return Ikcws

def decoder_7_4(cw_bits):
    """
    :param:
    cw_bits:  code word bits
    :return: 
    Ikhat: error corrected message estimate 
    """
    H = np.array(([1, 1, 1, 0, 1, 0, 0],
                  [0, 1, 1, 1, 0, 1, 0],
                  [1, 1, 0, 1, 0, 0, 1]))
    HT = np.transpose(H)
    r = cw_bits.copy()
    r = r.flatten().reshape(-1, 7)
    s = (r @ HT) % 2
    Ikhat = r.copy()
    for row in HT:
        Ikhat[(s==row).all(axis=1)] = ((r[(s==row).all(axis=1)] ^ [(HT==row).all(axis=1)]) % 2)
    Ikhat = Ikhat[:,0:4]
    Ikhat = Ikhat.flatten()
    return Ikhat

def encoder_15_11(Ik):
    """
    Inputs:
    Ik: Message Bits assume array length is integer divisble by 11
    Outputs
    Ikcws : Message bits in code words, first dimension is 1
    """
    G = np.array(([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]))
    Ik_new = Ik.copy()
    Ik_new = Ik_new.reshape(-1, 11)
    Ikcws = (Ik_new @ G) % 2
    Ikcws = Ikcws.flatten()
    return Ikcws

def decoder_15_11(cw_bits):
    """
    :param:
    cw_bits:  code word bits
    :return: 
    Ikhat: error corrected message estimate 
    """
    H = np.array(([1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                  [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
                  [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]))
    HT = np.transpose(H)
    r = cw_bits.copy()
    r = r.flatten().reshape(-1, 15)
    s = (r @ HT) % 2
    Ikhat = r.copy()
    for row in HT:
        Ikhat[(s==row).all(axis=1)] = ((r[(s==row).all(axis=1)] ^ [(HT==row).all(axis=1)]) % 2)
    Ikhat = Ikhat[:,0:11]
    Ikhat = Ikhat.flatten()
    return Ikhat

def encoder_31_26(Ik):
    """
    Inputs:
    Ik: Message Bits assume array length is integer divisble by 4
    Outputs
    Ikcws : Message bits in code words, first dimension is 1
    """
    G = np.array(([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0]))
    Ik_new = Ik.copy()
    Ik_new = Ik_new.reshape(-1, 26)
    Ikcws = (Ik_new @ G) % 2
    Ikcws = Ikcws.flatten()
    return Ikcws

def decoder_31_26(cw_bits):
    """
    :param:
    cw_bits:  code word bits
    :return: 
    Ikhat: error corrected message estimate 
    """
    H = np.array(([1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                  [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                  [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                  [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]))
    HT = np.transpose(H)
    r = cw_bits.copy()
    r = r.flatten().reshape(-1, 31)
    s = (r @ HT) % 2
    Ikhat = r.copy()
    for row in HT:
        Ikhat[(s==row).all(axis=1)] = ((r[(s==row).all(axis=1)] ^ [(HT==row).all(axis=1)]) % 2)
    Ikhat = Ikhat[:,0:26]
    Ikhat = Ikhat.flatten()
    return Ikhat