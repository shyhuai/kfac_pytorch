# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import reader
import utils as u

OUTPUT_PATH='./pdfresults/icdcs2021'

STANDARD_TITLES = {
        'resnet20': 'ResNet-20',
        'vgg16': 'VGG-16',
        'alexnet': 'AlexNet',
        'resnet34': 'ResNet-34',
        'resnet50': 'ResNet-50',
        'resnet152': 'ResNet-152',
        'densenet161': 'DenseNet-161',
        'densenet201': 'DenseNet-201',
        'inceptionv4': 'Inception-v4',
        'lstmptb': 'LSTM-PTB',
        'lstm': 'LSTM-PTB',
        'lstman4': 'LSTM-AN4'
        }


DNN_MARKERS = {
        'resnet152':'d',
        'resnet34':'*',
        'densenet201':'o',
        'resnet50':'^',
        }

DNN_COLORS = {
        'resnet152':'green',
        'resnet34':'red',
        'densenet201':'m',
        'resnet50':'black',
        }

def analyze_tensor_sizes():
    fig, ax = plt.subplots(figsize=(6,4.5))

    def _plot_dnn_tensor(dnn):
        fn = '/Users/lele/shared-server/kfac-logs/%s-matrixsize.log' % (dnn)
        sizes = reader.read_tensor_sizes(fn)
        sizes = [s[0]*(s[0]+1)//2 for s in sizes]
        print('dnn: ', dnn, ', min: %d, max: %d' % (np.min(sizes), np.max(sizes)))
        counter_dict = {}
        for s in sizes:
            if s not in counter_dict:
                counter_dict[s] = 0
            counter_dict[s] += 1
        keys = list(counter_dict.keys())
        keys.sort()
        print(dnn, 'sizes: ', keys)
        x_pos = [i for i, _ in enumerate(keys)]
        counters = [counter_dict[k] for k in keys]
        #print(dnn, 'counters: ', counters)
        #print(dnn, 'Total tensors: ', np.sum(counters))
        #ax2.bar(x_pos, counters, color='green')
        ax.scatter(np.array(keys)*4, counters, color=DNN_COLORS[dnn], marker=DNN_MARKERS[dnn], facecolors='none', linewidth=1, label=STANDARD_TITLES[dnn])
        #ax2.set_xticks(x_pos, keys)
        ax.set_xlabel('Tensor size (# of communicated elements)')
        ax.set_ylabel('Count')
        threshold = 128
        idx = 0
        for i, s in enumerate(keys):
            if s > threshold:
                idx = i
                break
        thres_count = np.sum(counters[0:idx])
        #print(dnn, 'counter smaller than threshold: ', thres_count)

    lines = []
    labels = []
    dnn='resnet34'
    _plot_dnn_tensor(dnn)
    dnn='resnet50'
    _plot_dnn_tensor(dnn)
    dnn='resnet152'
    _plot_dnn_tensor(dnn)
    lines, labels = ax.get_legend_handles_labels()

    #fig.legend(loc='upper center', ncol=3)
    plt.legend(ncol=1, loc=1, prop={'size': 14})
    u.update_fontsize(ax, 14)
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xscale('log')
    #plt.title(dnn)
    plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'tensordistribution'), bbox_inches='tight')
    #plt.show()

if __name__ == '__main__':
    analyze_tensor_sizes()


