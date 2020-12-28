from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import utils

OUTPUT_PATH='./pdfresults/icdcs2021'

class Color:
    backward_color = '#4d7d2e'
    comm_color = '#F1B183'
    factor_color = '#a4c2f4ff'
    factorcomm_color = '#b45f06ff'
    inverse_color = '#3c78d8ff'
    inversecomm_color = '#cc0000ff'


def plot_breakdown():
    FONTSIZE=12
    names = ['FF & BP', 'GradComm', 'FactorComp', 'FactorComm', 'InverseComp', 'InverseComm']
    colors = [Color.backward_color, Color.comm_color, Color.factor_color, Color.factorcomm_color, Color.inverse_color, Color.inversecomm_color]

    dnn='resnet50'
    sgd = [0.132,    0, 0,    0, 0, 0]
    ssgd = [0.132,    0.067, 0,    0, 0, 0]
    kfac = [0.132,    0, 0.205, 0,    0.282, 0]
    dkfac = [0.132,    0.199-0.132, 0.404-0.199, 0.704-0.404, 0.282, 0]
    dkfacmp = [0.132,    0.199-0.132, 0.404-0.199, 0.704-0.404, 0.736-0.704, 0.882-0.736]
    #names = ['FF & BP', 'Compression', 'Communication', 'LARS']
    #colors = [Color.backward_color, Color.compression_color, Color.comm_color, Color.lars_color]
    #densesgd = [0.204473, 0,    0.24177, 0.01114]
    #topksgd =  [0.204473,   0.239, 0.035, 0.01114]
    #densesgd96 = [0.054376, 0,    0.366886, 0.012794]
    #topksgd96 =  [0.054376,   0.239, 0.035, 0.012794]


    fig, ax = plt.subplots(figsize=(5.8,4))

    count = 5
    ind = np.arange(count)
    width = 0.8
    margin = 0.02
    xticklabels = ['SGD', 'S-SGD', 'KFAC', 'D-KFAC', 'D-KFAC-MP']
    newind = np.arange(count).astype(np.float32)
    bars = []
    bottom=np.array([0]*count).astype(np.float32)
    for i in range(len(sgd)):
        label = names[i]
        data=[sgd[i], ssgd[i], kfac[i], dkfac[i], dkfacmp[i]]
        p1 = ax.bar(newind, data, width, bottom=bottom, color=colors[i], label=label, edgecolor='black')
        bottom += np.array(data)
        bars.append(p1[0])
    #utils.autolabel(p1, ax, r'1 GPU', 0, 10)

    #newind += width+margin
    #bottom=0
    #for i in range(len(sgd)):
    #    label = names[i]
    #    data=[ssgd[i], dkfac[i]]
    #    p1 = ax.bar(newind, data, width, bottom=bottom, color=colors[i], label=label, edgecolor='black')
    #    bottom += np.array(data)
    #utils.autolabel(p1, ax, r'64 GPUs', 0, 10)
        #bars.append(p1[0])


    handles, labels = ax.get_legend_handles_labels()
    #ax.legend([handles[0][0]], [labels[0][0]], ncol=2)
    print(labels)
    print(handles)
    #ax.set_xlim(right=2.5)
    ax.set_ylim(top=ax.get_ylim()[1]*1.05)
    ax.set_xticks(newind)# -(width+margin)/2)
    ax.set_xticklabels(xticklabels, rotation=30)
    #ax.set_xlabel('Model')
    ax.set_ylabel('Time [s]')
    utils.update_fontsize(ax, FONTSIZE)
    ax.legend(tuple(bars), tuple(names), loc='center left',bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE)#, handletextpad=0.2, columnspacing =1.)
    #ax.legend(tuple(bars), tuple(names), loc='upper center',bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE, ncol=3)#, handletextpad=0.2, columnspacing =1.)
    fig.subplots_adjust(left=0.14, right=0.61, bottom=0.19, top=0.94)
    #plt.savefig('%s/naive-breakdown-%s.pdf' % (OUTPUT_PATH, dnn), bbox_inches='tight')
    plt.savefig('%s/naive-breakdown-%s.pdf' % (OUTPUT_PATH, dnn))
    #plt.show()


if __name__ == '__main__':
    plot_breakdown()
