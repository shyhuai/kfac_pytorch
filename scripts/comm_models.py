from __future__ import print_function
import numpy as np
import reader
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

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
    #fn='logs/nccl-bcast-n16IB.log'
    fn='logs/nccl-bcast-n64.log'
    sizes, comms, errors = reader.read_times_from_nccl_log(fn, original=True)
    print('sizes: ', sizes)
    print('comms: ', comms)
    print('errors: ', errors)
    alpha, beta = _fit_linear_function(np.array(sizes), comms)
    py = alpha + beta * np.array(sizes)

    fig, ax = plt.subplots(figsize=(8.,4))
    ax.plot(comms, marker='o', label='measured')
    ax.plot(py, marker='^', label='fit')
    plt.show()


if __name__ == '__main__':
    model_bcast_log()

