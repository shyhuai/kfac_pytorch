import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

def read_log(filename):
    f = open(filename, 'r')
    sizes = []
    computes = []
    comms = []
    merged_comms = []
    for l in f.readlines():
        items = l.split('[')[1][0:-2].split(',')
        items = [float(it.strip()) for it in items]
        if int(items[2]) == 0 or int(items[3]) == 0:# or int(items[1]) > 1000000:
            continue
        #sizes.append(float(items[1])*4)
        sizes.append(float(items[1]))
        computes.append(items[2])
        #comms.append(items[3])
        comms.append(float(items[4]))
        merged_comms.append(items[4])
    f.close()
    #print('filename: ', filename)
    #print('sizes: ', sizes)
    #print('total sizes: ', np.sum(sizes))
    #print('sizes len: ', len(sizes))
    #print('computes: ', computes)
    #print('communications: ', comms)
    return sizes, comms, computes, merged_comms

def read_p100_log(filename):
    f = open(filename, 'r')
    computes = []
    sizes = []
    for l in f.readlines():
        items = l.split(',') 
        sizes.append(float(items[-2]))
        computes.append(float(items[-1]))
        #comms.append(items[3])
    # remove duplicate
    reals = []
    realc = []
    pre = -1

    for i, comp in enumerate(computes):
        if pre != comp:
            reals.append(sizes[i])
            realc.append(comp)
        else:
            reals[-1] += sizes[i]
        pre = comp
    f.close()
    return sizes, realc


def to_percent(y, position):
    s = str(100 * y)
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def plot_hist(d, title=None, ax=None):
    d = np.array(d)
    flatten = d.ravel()
    mean = np.mean(flatten)
    std = np.std(flatten)
    if ax is None:
        ax = plt
    #norm = np.linalg.norm(flatten)
    #flatten = flatten[np.where(flatten!=0.0)]
    count, bins, ignored = plt.hist(flatten, 1000, normed=False, alpha=0.5, label=title)
    print('count: ', count)
    print('count sum: ', np.sum(count))
    print('bins: ', bins)
    print('mean: %f, std: %f' % (mean, std))
    print('min: %f' % np.min(np.abs(flatten)))
    n_neg = flatten[np.where(flatten==0.0)].size
    print('# of zero: %d' % n_neg)
    print('# of total: %d' % flatten.size)
    #return n_neg, flatten.size # return #negative, total
    ax.set_ylabel('Count')
    ax.set_xlabel('Gradient value')
    return flatten


def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

def autolabel(rects, ax, label, rotation=90, fontsize=12):
    """
    Attach a text label above each bar displaying its height
    """
    for i, rect in enumerate(rects):
        height = rect.get_y() + rect.get_height()
        if type(label) is list or type(label) is np.ndarray:
            l = label[i]
        else:
            l = label
        ax.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            l,
            ha='center', va='bottom', rotation=rotation, fontsize=fontsize)

def force_insert_item(d, key, val):
    if key not in d:
        d[key] = []
    d[key].append(val)

def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
