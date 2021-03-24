from __future__ import print_function
import matplotlib
import numpy as np
import reader
import matplotlib.pyplot as plt
import utils 

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

def model_bcast_log():
    FONTSIZE=18
    #plt.rc('font', size=FONTSIZE-4)
    #fn='logs/nccl-bcast-n16IB.log'
    #fn='logs/nccl-bcast-n64.log'
    #sizes, comms, errors = reader.read_times_from_nccl_log(fn, start=1024, end=1024*1024*512, original=True)
    comm_op = 'allreduce';short='ar'
    #comm_op = 'broadcast';short='bcast'
    #fn='logs/%s-n64-ib1-largesize.log' % comm_op
    #fn='logs/%s-n64-ib1-smallsize.log' % comm_op
    fn='logs/nccl-%s-n64.log' % comm_op
    #sizes, comms= reader.read_from_log_mean_std(fn)
    sizes, comms, errors = reader.read_times_from_nccl_log(fn, start=1024*1024, end=1024*1024*512, original=True)
    sizes = np.array(sizes)/4
    print('sizes: ', sizes)
    print('comms: ', comms)
    #print('errors: ', errors)
    alpha, beta = _fit_linear_function(np.array(sizes), comms)
    print('alpha: ', alpha, ', beta: ', beta)
    py = alpha + beta * np.array(sizes)

    fig, ax = plt.subplots(figsize=(4.4,4.5))
    #ax.plot(sizes, comms, marker='o', label='measured')
    #ax.plot(sizes, py, marker='^', label='fit')
    #fig, ax = plt.subplots()
    measured, = ax.plot(sizes, comms, label='Measured')
    #ax.plot(sizes, py, label=r'Predicted ($\alpha=%f')
    predicted, = ax.plot(sizes, py, '--', label=r'Predicted \n($\alpha_{%s}$=%.2e, $\beta_{%s}$=%.2e)'%(short, Decimal(alpha), short, Decimal(beta)))
    ax.set_xlabel('# of 32-bit elements')
    ax.set_ylabel('Communication time [s]')
    #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    #ax.legend(fontsize=FONTSIZE)
    ax.legend([measured, predicted], ['Measured', 'Predicted \n' + r'($\alpha_{%s}$=%.2e'%(short, Decimal(alpha)) +'\n'+ r'$\beta_{%s}$=%.2e)'% (short, Decimal(beta))], fontsize=FONTSIZE)
    utils.update_fontsize(ax, FONTSIZE)
    plt.subplots_adjust(left=0.20, bottom=0.14, top=0.99, right=0.99)
    mf = matplotlib.ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2,2))
    plt.gca().yaxis.set_major_formatter(mf)
    #plt.savefig('%s/%s-communicaion-model.pdf' % (OUTPUT_PATH, comm_op))
    plt.show()


if __name__ == '__main__':
    model_bcast_log()

