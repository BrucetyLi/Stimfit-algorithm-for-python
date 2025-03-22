# T2 mapping fitting with stimulated echo correction based on 'StimFit'.
# StimFit is described in the paper
# "R. Lebel, et al., StimFit: A toolbox for robust T2 mapping with stimulated echo compensation, ISMRM 2012".

# StimFit by R. Marc Lebel (https://mrel.usc.edu/sharing/)

# Copyright, Shaohang Li, Fudan University, China. All rights reserved.

import stimfit
import numpy as np
import scipy
import time


if __name__ == '__main__':
    """Please ensure the following packages are available before use.
    Numpy, scipy, numba(for acceleration), concurrent.futures(for multiprocessing)
    """
    # The following is an example of using pystimfit to process GE data.

    """ Data Input """
    # Normalized data should be input
    pic = scipy.io.loadmat(r'data\GE.mat')['GE']    # format: [Dy, Dx, echos].

    """Custom Options"""
    # Default Properties: 's':selective, 'n':non-selective; three vendors: 'GE', 'Philips', 'Siemens'
    opt = stimfit.initializer('s', 'GE')

    # The following options can be defined by users. If not, the default value will be used.
    # (The following values correspond to our GE data)
    opt['T1'] = 1.5
    opt['Dz'] = [0, 0.45]
    opt['esp'] = 12.9 / 1000
    opt['etl'] = 10
    opt['RFr']['angle'] = 180
    opt['RFr']['FA_array'] = np.ones(opt['etl'])

    """Two different ways of use"""
    time1 = time.time()
    # method1: Pixel by pixel
    # T2map = np.zeros(pic.shape[:-1])
    # B1 = np.zeros(pic.shape[:-1])
    # amp = np.zeros(pic.shape[:-1])
    # for i in range(T2map.shape[0]):
    #     for j in range(T2map.shape[1]):
    #         [T2map[i,j],amp[i,j],B1[i,j],opt] = stimfit.fit(pic[i,j,:],opt)

    # method2: Multiprocess for acceleration
    T2map, amp, B1, opt = stimfit.multiprocess_fit(pic, opt)
    print(time.time()-time1)

    """Saving data"""
    np.save(r'results\sGE.npy', np.array([T2map, amp, B1]))



