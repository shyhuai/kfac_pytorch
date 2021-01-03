from __future__ import print_function
import matplotlib
import numpy as np
import reader
import matplotlib.pyplot as plt
import utils 
import math

from scipy.optimize import curve_fit
from decimal import Decimal
from sklearn.linear_model import LinearRegression

OUTPUT_PATH='./pdfresults/icdcs2021'

def _fit_linear_function(x, y):
    X = np.array(x).reshape((-1, 1))
    Y = np.array(y)
    print('x: ', X)
    print('y: ', Y)
    model = LinearRegression()
    model.fit(X, Y)
    alpha = model.intercept_
    beta = model.coef_[0]
    #A = np.vstack([X, np.ones(len(X))]).T
    #beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
    return alpha, beta

def func_fit(x, a, b):
    return a * np.exp(b * x)

def _get_x_y(gpu):
    fn = 'logs/inverse-%s.log' % gpu
    x, y = reader.read_tensorsize_vs_time(fn)
    x = np.array(x)
    y = np.array(y)
    return x, y
def _fit(x, y):
    curve = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
    a = np.exp(curve[1])
    b = curve[0]
    fitted = a * np.exp(b * x)
    return a, b, fitted

def model_inverse_compute_log():
    #fn = 'logs/inverse-resnet50.log'
    FONTSIZE=14
    v100='v100'
    rtx2080ti='rtx2080ti'
    x, y = _get_x_y(rtx2080ti)
    a, b, fitted = _fit(x, y)
    print(a, b)
    x1, y1 = _get_x_y(v100)
    a1, b1, fitted1 = _fit(x1, y1)

    fig, ax = plt.subplots(figsize=(8.,4))
    ax.plot(x, y, label='Measured (RTX2080Ti)')
    ax.plot(x, fitted, '--', label=r'Fitted (RTX2080Ti): $\alpha_{inv}=%.2e, \beta_{inv}=%.2e$' % (Decimal(a), Decimal(b)))

    ax.plot(x1, y1, label='Measured (V100)')
    ax.plot(x1, fitted1, '--', label=r'Fitted (V100): $\alpha_{inv}=%.2e, \beta_{inv}=%.2e$' % (Decimal(a), Decimal(b)))

    ax.set_xlabel('Height or width of a symmetric matrix')
    ax.set_ylabel('Inverse computation time [s]')
    utils.update_fontsize(ax, FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    #plt.savefig('%s/inverse-compute-model.pdf' % (OUTPUT_PATH), bbox_inches='tight')
    plt.show()

def compute_and_communication():
    FONTSIZE=14
    fig, ax = plt.subplots(figsize=(8.,4))

    comm_op = 'broadcast';short='bcast'
    fn='logs/%s-n64-ib1-largesize.log' % comm_op
    sizes, comms= reader.read_from_log_mean_std(fn)
    alpha, beta = _fit_linear_function(np.array(sizes), comms)

    fn = 'logs/inverse-rtx2080ti.log'
    sizes1, comps  = reader.read_tensorsize_vs_time(fn)
    sizes1 = np.array(sizes1)
    comps = np.array(comps)
    a1, b1, fitted1 = _fit(sizes1, comps)

    sym_sizes = np.array([m*(m+1)/2 for m in sizes1])
    prefit_comms = alpha + beta * sym_sizes

    ax.plot(sizes1, fitted1, label='Inverse computation time', color='#3c78d8ff')
    ax.plot(sizes1, prefit_comms, label='Symtric matrix communication time', color='#cc0000ff')

    ax.set_xlabel('Height or width of a symmetric matrix')
    ax.set_ylabel('Time [s]')
    utils.update_fontsize(ax, FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    plt.savefig('%s/inv-vs-bcast.pdf' % (OUTPUT_PATH), bbox_inches='tight')
    #plt.show()



if __name__ == '__main__':
    #model_inverse_compute_log()
    compute_and_communication()
