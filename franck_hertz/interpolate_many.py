import os
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# Make plots readable
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=8)
mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)
mpl.rc('lines', markersize=2)
plt.ion()

def norm(A):
    return (A-np.min(A))/(np.max(A)-np.min(A))

def myplot(*args, ax=None, title='', xlabel='Accelerating Voltage (V)', \
        ylabel='Arbitary', xlim=[0,80], ylim=[0,1], xdv=5, legend=[]):
    if ax is None:
        plt.clf()
        plt.figure()
        k = plt.plot(*args)
        plt.grid(True)
        if legend != []:
            plt.legend(legend)
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])
        plt.xticks(np.arange(xlim[0],xlim[1]+xdv, xdv))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
    else:
        ax.clear()
        k = ax.plot(*args)
        ax.grid(True)
        if legend != []:
            ax.legend(legend)
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])
        if xdv is not None: ax.set_xticks(np.arange(xlim[0],xlim[1]+xdv, xdv))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    return k

def ss(string):
    sum = 0
    for s in string:
        sum += ord(s)
    return sum

mypath = 'data/good/'

files = sorted([f for f in os.listdir(mypath) if (os.path.isfile(os.path.join(mypath, f)) and '.csv' in f)], key=ss)

rows, cols = int(np.ceil(np.sqrt(len(files)))), int(np.floor(np.sqrt(len(files))))
fig, axs = plt.subplots(rows, cols, figsize=(16, 9))
# plt.subplots_adjust(left=0.05,right=0.98,bottom=0.05,top=0.95)


for f in range(len(files)):
    # print(files[f])
    file_name = os.path.join(mypath, files[f])

    data = pd.read_csv(file_name).to_numpy()[1:,:].astype(float)
    data = data[data[:,0].argsort()]

    s, k = 0, 0
    vi, vo = np.array([]), np.array([])
    for i in range(1, len(data)):
        s+=data[i,1]
        k+=1
        if data[i, 0] != data[i-1, 0]:
            vo = np.append(vo, s/k)
            vi = np.append(vi, data[i, 0]*10)
            s,k=0,0
    
    idx = np.logical_and(vi>5,vi<80)
    vi, vo = vi[idx], norm(vo[idx])
    v2 = np.linspace(min(vi), max(vi), 1000)

    spl = UnivariateSpline(vi, vo, k=4, s=0.02)
    # print((int(np.floor(f/cols)),f%cols))
    p0 = myplot(vi, vo, "o", v2, spl(v2), ax=axs[int(np.floor(f/cols)),f%cols], xlim=[0,max(vi)+max(vi)%5], legend=["Data", "Spline"], title=file_name)

    dspl = spl.derivative()
    ddspl = dspl.derivative()
    roots = np.array([r for r in dspl.roots() if ddspl(r) > 0 and r > 28 and r < 80])#ddspl(r) > 0.01 and ddspl(r) < 0.2 and 

    # print(roots)
    # print(np.diff(roots))
    dr_cut = np.array([dr for dr in np.diff(roots) if dr < 5.5 and dr > 4.1])
    if len(dr_cut)>0:
        mean, std = np.mean(dr_cut), np.std(dr_cut)
        print(f"{file_name} | {mean} | {std}")
        for r in roots:
            axs[int(np.floor(f/cols)),f%cols].axvline(x=r, linestyle='--', color='r', alpha=0.3)

plt.savefig("manyplot.png")
input("Press Enter to continue...")