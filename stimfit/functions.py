import numpy as np
from numba import jit
from scipy import optimize, io
from concurrent.futures import ProcessPoolExecutor


def initializer(mode='n', vendor='GE'):
    opt = dict()
    if mode != 'n' and mode != 's':
        mode = 'n'
    opt['mode'] = mode
    opt['esp'] = 10e-3
    opt['etl'] = 20
    opt['T1'] = 3

    opt['RFe'] = dict()
    opt['RFr'] = dict()
    if opt['mode'] == 's':
        opt['Dz'] = [-0.5, 0.5]
        opt['Nz'] = 51
        opt['Nrf'] = 64
        opt['RFe'] = {'RF': [], 'tau': 2e-3, 'G': 0.5, 'phase': 0, 'ref': 1, 'alpha': []}
        opt['RFr'] = {'RF': [], 'tau': 2e-3, 'G': 0.5, 'phase': 90, 'ref': 0, 'alpha': []}
        if vendor == 'Siemens':
            opt['RFe']['tau'] = 3072/1e6
            opt['RFe']['G'] = 0.417
            opt['RFr']['tau'] = 3000/1e6
            opt['RFr']['G'] = 0.326
            RF90 = io.loadmat(r'data\rf90_GE.mat')['rf90']
            RF180 = io.loadmat(r'data\rf180_GE.mat')['rf180']
            opt['RFe']['RF'] = RF90.ravel()
            opt['RFr']['RF'] = RF180.ravel()

        elif vendor == 'Philips':
            opt['RFe']['tau'] = 3820 / 1e6
            opt['RFe']['G'] = 0.392
            opt['RFr']['tau'] = 6010 / 1e6
            opt['RFr']['G'] = 0.327
            RF90 = io.loadmat(r'data\rf90_philips.mat')['rf90']
            RF180 = io.loadmat(r'data\rf180_philips.mat')['rf180']
            opt['RFe']['RF'] = RF90.ravel()
            opt['RFr']['RF'] = RF180.ravel()

        else:
            opt['RFe']['tau'] = 2000 / 1e6
            opt['RFe']['G'] = 0.751599
            opt['RFr']['tau'] = 3136 / 1e6
            opt['RFr']['G'] = 0.276839
            RF90 = io.loadmat(r'data\rf90_GE.mat')['rf90']
            RF180 = io.loadmat(r'data\rf180_GE.mat')['rf180']
            opt['RFe']['RF'] = RF90.ravel()
            opt['RFr']['RF'] = RF180.ravel()

    opt['RFe']['angle'] = 90
    opt['RFr']['angle'] = 180
    opt['RFr']['FA_array'] = np.ones(opt['etl'])

    opt['lsq'] = {'Ncomp': 1}
    opt['lsq']['Icomp'] = {'X0': [0.06, 0.1, 1], 'XU': [3, 1e+3, 1.8], 'XL': [0.015, 0, 0.2]}  # [T2(s),amp,B1]
    opt['lsq']['IIcomp'] = {'X0': [0.02, 0.1, 0.131, 0.1, 1], 'XU': [0.25, 1e+3, 3, 1e+3, 1.8],
                            'XL': [0.015, 0, 0.25, 0, 0.2]}
    opt['lsq']['IIIcomp'] = {'X0': [0.02, 0.1, 0.036, 0.1, 0.131, 0.1, 1], 'XU': [0.035, 1e+3, 0.13, 1e3, 3, 1e+3, 1.8],
                             'XL': [0.015, 0, 0.035, 0, 0.13, 0, 0.2]}
    opt['lsq']['fopt'] = {'xtol': 5e-4, 'ftol': 1e-9}
    return opt


def setRF(RF, Nrf, Dz, Nz):
    gamma = 2 * np.pi * 42.575e6 / 10000  # GAUSS
    z = np.linspace(Dz[0], Dz[1], Nz)
    scale = RF['angle'] / (gamma * RF['tau'] * abs(np.sum(RF['RF'])) / len(RF['RF']) * 180 / np.pi)
    RF['RF'] = scale * RF['RF']

    M = np.zeros([3, Nz])
    M[2, :] = 1
    RF['RF'] = 1e-4 * RF['RF']  # approximation for
    # small tip angle

    phi = gamma * RF['G'] * z * RF['tau'] / Nrf
    cphi, sphi = np.cos(phi), np.sin(phi)
    cpRF, spRF = np.cos(RF['phase'] * np.pi / 180), np.sin(RF['phase'] * np.pi / 180)
    thetaRF = gamma * RF['RF'] * RF['tau'] / Nrf
    ctRF, stRF = np.cos(thetaRF), np.sin(thetaRF)

    for i in range(Nrf):
        for j in range(Nz):
            Rz = np.array([[cphi[j], sphi[j], 0], [-sphi[j], cphi[j], 0], [0, 0, 1]])
            M[:, j] = np.dot(Rz, M[:, j])

        R = np.array([[1, 0, 0], [0, ctRF[i], stRF[i]], [0, -stRF[i], ctRF[i]]])
        if RF['phase'] != 0:
            Rz = np.array([[cpRF, spRF, 0], [-spRF, cpRF, 0], [0, 0, 1]])
            Rzm = np.array([[cpRF, -spRF, 0], [spRF, cpRF, 0], [0, 0, 1]])
            R = np.dot(Rzm, np.dot(R, Rz))
        M = np.dot(R, M)

    if RF['ref'] > 0:
        psi = -RF['ref'] / 2 * gamma * RF['G'] * z * RF['tau']
        for j in range(Nz):
            Rz = np.array([[np.cos(psi[j]), np.sin(psi[j]), 0], [-np.sin(psi[j]), np.cos(psi[j]), 0], [0, 0, 1]])
            M[:, j] = np.dot(Rz, M[:, j])

    RF['RF'] = 1e4 * RF['RF']
    RF['alpha'] = 1e4 * np.arccos(M[2, :])
    return RF


@jit(nopython=True)
def epg(x2, b1, x1, esp, ar, ae):  # TE = 6.425ms. TR = 1500ms.   90,175,145,110,110,110.
    echo_intensity = np.zeros(ar.shape, dtype=np.float64)
    omiga = np.zeros((ar.shape[0], 3, 1 + 2 * ar.shape[1]), dtype=np.float64)
    ar = b1 * ar
    ae = b1 * ae
    x2 = np.exp(-0.5 * esp / x2)
    x1 = np.exp(-0.5 * esp / x1)

    for i in range(omiga.shape[2]):
        if i == 0:
            omiga[:, 0, i] = np.sin(ae)
            omiga[:, 1, i] = np.sin(ae)
            omiga[:, 2, i] = np.cos(ae)
            continue
        omiga[:, 0, 1:i + 1] = omiga[:, 0, 0:i]
        omiga[:, 1, 0:i] = omiga[:, 1, 1:i + 1]
        omiga[:, 0, 0] = np.conj(omiga[:, 1, 0])
        omiga[:, 0:2, :] = x2 * omiga[:, 0:2, :]
        omiga[:, 2, :] = x1 * omiga[:, 2, :]
        omiga[:, 2, 0] += 1 - x1
        if i % 2 == 1:
            for runs in range(ar.shape[0]):
                ari = ar[runs, i // 2]
                T = np.array(
                    [[np.cos(0.5 * ari) ** 2, np.sin(0.5 * ari) ** 2, np.sin(ari)],
                     [np.sin(0.5 * ari) ** 2, np.cos(0.5 * ari) ** 2, -np.sin(ari)],
                     [-0.5 * np.sin(ari), +0.5 * np.sin(ari), np.cos(ari)]], dtype=np.float64)
                omiga[runs, :, :] = np.dot(T, np.ascontiguousarray(omiga[runs, :, :]))
        if i % 2 == 0:
            echo_intensity[:, i // 2 - 1] = omiga[:, 0, 0]
    return echo_intensity


def epgsig(t2, b1, opt):
    sig = np.zeros(opt['etl'])
    if opt['mode'] == 'n':
        FA = np.pi / 180 * opt['RFr']['angle'] * np.array([opt['RFr']['FA_array']])
        sig = epg(t2, b1, opt['T1'], opt['esp'], FA, opt['RFe']['angle'] * np.pi / 180)
    elif opt['mode'] == 's':
        FA = np.array([opt['RFr']['alpha']]).T * opt['RFr']['FA_array']
        M = epg(t2, b1, opt['T1'], opt['esp'], FA, opt['RFe']['alpha'])
        sig = np.sum(M, 0) / opt['Nz']
    return sig.ravel()


def residual1(p, y, opt):
    return y - epgsig(p[0], p[2], opt) * p[1]


def residual2(p, y, opt):
    return y - epgsig(p[0], p[4], opt) * p[1] - epgsig(p[2], p[4], opt) * p[3]


def residual3(p, y, opt):
    return y - epgsig(p[0], p[6], opt) * p[1] - epgsig(p[4], p[6], opt) * p[5] - epgsig(p[2], p[6], opt) * p[3]


def fit(sig, opt, switch=1):
    if len(sig) != opt['etl']:
        raise Exception('Inconsistent echo train length')
    if opt['mode'] == 's' and switch == 1:
        opt['RFe'] = setRF(opt['RFe'], opt['Nrf'], opt['Dz'], opt['Nz'])
        opt['RFr'] = setRF(opt['RFr'], opt['Nrf'], opt['Dz'], opt['Nz'])

    if opt['lsq']['Ncomp'] == 2:
        X = optimize.least_squares(residual2, opt['lsq']['IIcomp']['X0'], args=(sig, opt),
                                   bounds=(opt['lsq']['IIcomp']['XL'], opt['lsq']['IIcomp']['XU']),
                                   xtol=opt['lsq']['fopt']['xtol'], ftol=opt['lsq']['fopt']['ftol']).x
        T2, amp, B1 = [X[0], X[2]], [X[1], X[3]], X[5]

    elif opt['lsq']['Ncomp'] == 3:
        X = optimize.least_squares(residual2, opt['lsq']['IIIcomp']['X0'], args=(sig, opt),
                                   bounds=(opt['lsq']['IIIcomp']['XL'], opt['lsq']['IIIcomp']['XU']),
                                   xtol=opt['lsq']['fopt']['xtol'], ftol=opt['lsq']['fopt']['ftol']).x
        T2, amp, B1 = [X[0], X[2], X[4]], [X[1], X[3], X[5]], X[6]

    else:
        X = optimize.least_squares(residual1, opt['lsq']['Icomp']['X0'], args=(sig, opt),
                                   bounds=(opt['lsq']['Icomp']['XL'], opt['lsq']['Icomp']['XU']),
                                   xtol=opt['lsq']['fopt']['xtol'], ftol=opt['lsq']['fopt']['ftol']).x
        T2, amp, B1 = X
    return T2, amp, B1, opt


def fit_help(z):
    return fit(z[0], z[1])


def multiprocess_fit(pic, opt):
    with ProcessPoolExecutor() as executors:
        results = list(executors.map(fit_help, [(pic[i // pic.shape[1], i % pic.shape[1], :], opt) for i in
                                                range(pic.shape[0] * pic.shape[1])]))
    T2map = np.array([results[i][0] for i in range(pic.shape[0] * pic.shape[1])],
                     dtype=np.float64).reshape(pic.shape[0], pic.shape[1])
    amp = np.array([results[i][1] for i in range(pic.shape[0] * pic.shape[1])],
                   dtype=np.float64).reshape(pic.shape[0], pic.shape[1])
    B1 = np.array([results[i][2] for i in range(pic.shape[0] * pic.shape[1])],
                  dtype=np.float64).reshape(pic.shape[0], pic.shape[1])
    opt = list(results)[0][3]
    return T2map, amp, B1, opt
