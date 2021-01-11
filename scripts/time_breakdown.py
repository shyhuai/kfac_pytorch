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


def plot_breakdown_naive():
    FONTSIZE=12
    names = ['FF & BP', 'GradComm', 'FactorComp', 'FactorComm', 'InverseComp', 'InverseComm']
    colors = [Color.backward_color, Color.comm_color, Color.factor_color, Color.factorcomm_color, Color.inverse_color, Color.inversecomm_color]

    dnn='resnet50'
    sgd = [0.132,    0, 0,    0, 0, 0]
    ssgd = [0.132,    0.1968-0.132, 0,    0, 0, 0]
    kfac = [0.132,    0, 0.4058-0.1968, 0,    0.8706-0.5783, 0]
    dkfac = [0.132,    0.1968-0.132, 0.4058-0.1968, 0.5783-0.4058, 0.8706-0.5783, 0]
    dkfacmp = [0.132,    0.1968-0.132, 0.4058-0.1968, 0.5783-0.4058, 0.6295-0.5783, 0.7635-0.6295]

    fig, ax = plt.subplots(figsize=(5.8,4))

    count = 5
    ind = np.arange(count)
    width = 0.8
    margin = 0.02
    xticklabels = ['SGD', 'S-SGD', 'KFAC', 'D-KFAC', 'MPD-KFAC']
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


def plot_breakdown_spdkfac():
    FONTSIZE=12
    names = ['FF & BP', 'GradComm', 'FactorComp', 'FactorComm', 'InverseComp', 'InverseComm']
    colors = [Color.backward_color, Color.comm_color, Color.factor_color, Color.factorcomm_color, Color.inverse_color, Color.inversecomm_color]

    dnn='resnet50'
    sgd = [0.132,    0, 0,    0, 0, 0]
    ssgd = [0.132,    0.067, 0,    0, 0, 0]
    kfac = [0.132,    0, 0.205, 0,    0.282, 0]
    dkfac = [0.132,    0.199-0.132, 0.404-0.199, 0.704-0.404, 0.282, 0]
    dkfacmp = [0.132,    0.199-0.132, 0.404-0.199, 0.704-0.404, 0.736-0.704, 0.882-0.736]

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
    #plt.savefig('%s/naive-breakdown-%s.pdf' % (OUTPUT_PATH, dnn))
    plt.show()


def plot_breakdown_spdkfac():
    fig, ax = plt.subplots(figsize=(7.0,4.4))
    FONTSIZE=12

    names = ['FF & BP', 'GradComm', 'FactorComp', 'FactorComm', 'InverseComp', 'InverseComm']
    colors = [Color.backward_color, Color.comm_color, Color.factor_color, Color.factorcomm_color, Color.inverse_color, Color.inversecomm_color]

    xticklabels = ['ResNet-50', 'ResNet-152', 'DenseNet-201', 'Inception-v4']
    dnns = ['resnet50', 'resnet152', 'densenet201', 'inceptionv4']
    #algos = ['dkfac', 'dkfac-mp', 'spd-kfac']
    algos = ['dkfac', 'mpd-kfac', 'spd-kfac']
    labels=['D-K.', 'MPD-K.', 'SPD-K.']
    data = {'resnet50':  # [compute, communicate gradient, compute factor, communicate factor, compute inverse, communicate inverse]
                    {
                     'dkfac':    [0.132, 0.1968, 0.4083, 0.5783, 0.8525, 0.8525],
                     'mpd-kfac': [0.132, 0.1968, 0.4083, 0.5783, 0.6295, 0.7635],
                     'spd-kfac': [0.132, 0.1968, 0.4083, 0.5064, 0.6114, 0.6755],
                     }, 
            'resnet152':  {
                     'dkfac':    [0.1140, 0.2730, 0.4657, 0.9048, 1.5807, 1.5807],
                     'mpd-kfac': [0.1140, 0.2730, 0.4657, 0.9016, 0.9555, 1.3933], 
                     'spd-kfac': [0.1140, 0.2730, 0.4657, 0.7417, 1.0231, 1.1689],
                     }, 
            'densenet201':  {
                     'dkfac':    [0.178, 0.3643, 0.6829, 1.0146, 1.4964, 1.4964],
                     'mpd-kfac': [0.178, 0.3643, 0.6806, 1.0308, 1.0660, 1.5340],
                     'spd-kfac': [0.178, 0.3643, 0.6806, 0.9243, 1.3266, 1.3615], 
                     }, 
            'inceptionv4':  {
                     'dkfac':    [0.134, 0.2669, 0.4648, 0.7551, 1.1857, 1.1857],
                     'mpd-kfac': [0.134, 0.2669, 0.4597, 0.7547, 0.8034, 1.1473],
                     'spd-kfac': [0.134, 0.2669, 0.4597, 0.6635, 0.9174, 0.9907],
                     }, 

            }
    def Smax(times):
        tf = times[0]; tb=times[1]; tc=times[2]
        r = tc/tb
        s = 1+1.0/(tf/min(tc,tb)+max(r,1./r))
        return s
    count = len(dnns)
    width = 0.2; margin = 0.02
    s = (1 - (width*count+(count-1) *margin))/2+width
    ind = np.array([s+i+1 for i in range(count)])
    
    for ia, algo in enumerate(algos):
        newind = ind+s*width+(s+1)*margin
        bp = []; gradcomm=[];factorcomp=[];factorcomm=[]; inversecomp=[]; inversecomm=[]
        one_group = [[] for i in range(len(names))]
        for dnn in dnns:
            d = data[dnn]
            ald = d[algo]
            t0 = 0.0
            for j, t in enumerate(ald):
                one_group[j].append(t-t0)
                t0 = t
        legend_p = []
        bottom = np.array([0.0]*len(one_group[0]))
        for k, d in enumerate(one_group):
            color = colors[k]
            label = names[k]
            p = ax.bar(newind, d, width, bottom=bottom, color=color,edgecolor='black', label=label)
            legend_p.append(p[0])
            bottom += np.array(d)
        s += 1 
        #ax.text(4, 4, 'ehhlo', color='b')
        utils.autolabel(p, ax, labels[ia], 90, FONTSIZE-2)
    ax.set_ylim(top=ax.get_ylim()[1]*1.15)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(legend_p, names, ncol=3, handletextpad=0.2, columnspacing =1., loc='upper center', fontsize=FONTSIZE, bbox_to_anchor=[0.5, 1.2])
    ax.set_ylabel('Time [s]')
    #ax.set_xticks(newind-width-margin/2)
    #ax.set_xticks(newind-width/2-margin/2)
    ax.set_xticks(newind-width*2/2-margin*2/2)
    ax.set_xticklabels(xticklabels)
    utils.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/spdkfac-vs-mpd-fac-timebreakdown.pdf' % (OUTPUT_PATH), bbox_inches='tight')
    #plt.show()


def plot_breakdown_pipelining():
    fig, ax = plt.subplots(figsize=(7.0,4.4))
    FONTSIZE=12

    names = ['FactorComp', 'FactorComm']
    colors = [Color.factor_color, Color.factorcomm_color]

    xticklabels = ['ResNet-50', 'ResNet-152', 'DenseNet-201', 'Inception-v4']
    dnns = ['resnet50', 'resnet152', 'densenet201', 'inceptionv4']
    #algos = ['dkfac', 'dkfac-mp', 'spd-kfac']
    #algos = ['mpd-kfac', 'spd-kfac']
    algos = ['mpd-kfac', 'lw-wo-tf', 'lw-wi-ttf', 'sp-wi-otf']
    labels=['Naive', 'LW w/o TF', 'LW w/ TTF', 'SP w/ OTF']
    data = {'resnet50':  
                    {
                     'mpd-kfac':  [0.2115, 0.3814],
                     'lw-wo-tf':  [0.2115, 0.4174],
                     'lw-wi-ttf': [0.2115, 0.3401],
                     'sp-wi-otf': [0.2115, 0.3096],
                     }, 
            'resnet152':  {
                     'mpd-kfac':  [0.1927, 0.6285],
                     'lw-wo-tf':  [0.1927, 0.7158],
                     'lw-wi-ttf': [0.1927, 0.5371],
                     'sp-wi-otf': [0.1927, 0.4687],
                     }, 
            'densenet201':  {
                     'mpd-kfac':  [0.3163, 0.6665],
                     'lw-wo-tf':  [0.3163, 0.7714],
                     'lw-wi-ttf': [0.3163, 0.5841],
                     'sp-wi-otf': [0.3163, 0.5600],
                     }, 
            'inceptionv4':  {
                     'mpd-kfac':  [0.1979, 0.4882],
                     'lw-wo-tf':  [0.1979, 0.6683],
                     'lw-wi-ttf': [0.1979, 0.4115],
                     'sp-wi-otf': [0.1979, 0.3967],
                     }, 

            }
    count = len(dnns)
    width = 0.2; margin = 0.02
    s = (1 - (width*count+(count-1) *margin))/2+width
    ind = np.array([s+i+1 for i in range(count)])
    
    for i, algo in enumerate(algos):
        newind = ind+s*width+(s+1)*margin
        bp = []; gradcomm=[];factorcomp=[];factorcomm=[]; inversecomp=[]; inversecomm=[]
        one_group = [[] for ii in range(len(names))]
        for dnn in dnns:
            d = data[dnn]
            ald = d[algo]
            t0 = 0.0
            for j, t in enumerate(ald):
                one_group[j].append(t-t0)
                t0 = t
        legend_p = []
        bottom = np.array([0.0]*len(one_group[0]))
        for k, d in enumerate(one_group):
            color = colors[k]
            label = names[k]
            p = ax.bar(newind, d, width, bottom=bottom, color=color,edgecolor='black', label=label)
            legend_p.append(p[0])
            bottom += np.array(d)
        s += 1 
        #ax.text(4, 4, 'ehhlo', color='b')
        utils.autolabel(p, ax, labels[i], 90, FONTSIZE-2)
    ax.set_ylim(top=ax.get_ylim()[1]*1.25)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(legend_p, names, ncol=1, handletextpad=0.2, columnspacing =1., loc='upper left', fontsize=FONTSIZE)
    ax.set_ylabel('Time [s]')
    #ax.set_xticks(newind-width-margin/2)
    ax.set_xticks(newind-width*3/2-margin*3/2)
    ax.set_xticklabels(xticklabels)
    utils.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/pipelining-timebreakdown.pdf' % (OUTPUT_PATH), bbox_inches='tight')
    #plt.show()


def plot_breakdown_bwp():
    fig, ax = plt.subplots(figsize=(7.0,4.4))
    FONTSIZE=12

    names = ['InverseComp', 'InverseComm']
    colors = [Color.inverse_color, Color.inversecomm_color]

    xticklabels = ['ResNet-50', 'ResNet-152', 'DenseNet-201', 'Inception-v4']
    dnns = ['resnet50', 'resnet152', 'densenet201', 'inceptionv4']
    #algos = ['dkfac', 'dkfac-mp', 'spd-kfac']
    #algos = ['mpd-kfac', 'spd-kfac']
    algos = ['algo1', 'algo2', 'algo3']
    labels=['Non-Dist', 'Seq-Dist', 'LBP']
    data = {'resnet50':  
                    {
                     'algo1':  [0.2742, 0.2742],
                     'algo2':  [0.0512, 0.1852],
                     'algo3':  [0.1049, 0.1691],
                     }, 
            'resnet152':  {
                     'algo1':  [0.6759, 0.6759],
                     'algo2':  [0.0539, 0.4917],
                     'algo3':  [0.2814, 0.4271],
                     }, 
            'densenet201':  {
                     'algo1':  [0.4818, 0.4818],
                     'algo2':  [0.0352, 0.5032],
                     'algo3':  [0.4023, 0.4373],
                     }, 
            'inceptionv4':  {
                     'algo1':  [0.4306, 0.4306],
                     'algo2':  [0.0487, 0.3926],
                     'algo3':  [0.2539, 0.3272],
                     }, 

            }
    count = len(dnns)
    width = 0.2; margin = 0.02
    s = (1 - (width*count+(count-1) *margin))/2+width
    ind = np.array([s+i+1 for i in range(count)])
    
    for i, algo in enumerate(algos):
        newind = ind+s*width+(s+1)*margin
        bp = []; gradcomm=[];factorcomp=[];factorcomm=[]; inversecomp=[]; inversecomm=[]
        one_group = [[] for ii in range(len(names))]
        for dnn in dnns:
            d = data[dnn]
            ald = d[algo]
            t0 = 0.0
            for j, t in enumerate(ald):
                one_group[j].append(t-t0)
                t0 = t
        legend_p = []
        bottom = np.array([0.0]*len(one_group[0]))
        for k, d in enumerate(one_group):
            color = colors[k]
            label = names[k]
            p = ax.bar(newind, d, width, bottom=bottom, color=color,edgecolor='black', label=label)
            legend_p.append(p[0])
            bottom += np.array(d)
        s += 1 
        #ax.text(4, 4, 'ehhlo', color='b')
        utils.autolabel(p, ax, labels[i], 90, FONTSIZE-2)
    ax.set_ylim(top=ax.get_ylim()[1]*1.25)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(legend_p, names, ncol=1, handletextpad=0.2, columnspacing =1., loc='upper right', fontsize=FONTSIZE)
    ax.set_ylabel('Time [s]')
    #ax.set_xticks(newind-width-margin/2)
    ax.set_xticks(newind-width*2/2-margin*2/2)
    ax.set_xticklabels(xticklabels)
    utils.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/bwp-timebreakdown.pdf' % (OUTPUT_PATH), bbox_inches='tight')
    #plt.show()

def plot_breakdown_stepbystep():
    fig, ax = plt.subplots(figsize=(7.4,4.4))
    FONTSIZE=12


    xticklabels = ['ResNet-50', 'ResNet-152', 'DenseNet-201', 'Inception-v4']
    dnns = ['resnet50', 'resnet152', 'densenet201', 'inceptionv4']
    #algos = ['dkfac', 'dkfac-mp', 'spd-kfac']
    #algos = ['mpd-kfac', 'spd-kfac']
    algos = ['algo1', 'algo2', 'algo3', 'algo4']
    labels=['-Pipe-LBP', '+Pipe-LBP', '-Pipe+LBP', '+Pipe+LBP']
    names = labels
    colors = ['white', Color.factorcomm_color, Color.inversecomm_color, 'black']
    resnet50 = [0.8525, 0.7806, 0.7474, 0.6755]
    resnet152 = [1.5807, 1.4176, 1.3319, 1.1689]
    densenet201=[1.4964, 1.4061, 1.4519, 1.3615]
    inceptionv4=[1.1857, 1.0941, 1.0823, 0.9907]

    count = len(dnns)
    width = 0.2; margin = 0.02
    s = (1 - (width*count+(count-1) *margin))/2+width
    ind = np.array([s+i+1 for i in range(count)])
    
    legend_p = []
    for i, algo in enumerate(algos):
        newind = ind+s*width+(s+1)*margin
        bp = [resnet50[i], resnet152[i], densenet201[i], inceptionv4[i]]
        color = colors[i]
        label = names[i]
        p = ax.bar(newind, bp, width, color=color,edgecolor='black', label=label)
        legend_p.append(p[0])
        s += 1 
    ax.set_ylim(bottom=0.6)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(legend_p, names, ncol=1, handletextpad=0.2, columnspacing =1., loc='upper left', fontsize=FONTSIZE)
    ax.set_ylabel('Time [s]')
    #ax.set_xticks(newind-width-margin/2)
    ax.set_xticks(newind-width*3/2-margin*3/2)
    ax.set_xticklabels(xticklabels)
    utils.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/step-by-step.pdf' % (OUTPUT_PATH), bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':
    #plot_breakdown_naive()
    plot_breakdown_spdkfac()
    #plot_breakdown_pipelining()
    #plot_breakdown_bwp()
    #plot_breakdown_stepbystep()
