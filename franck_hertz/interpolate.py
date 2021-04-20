import os
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# Make plots readable
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=24)
# mpl.rc('xtick', labelsize=20)
# mpl.rc('ytick', labelsize=20)
# mpl.rc('lines', markersize=5)
# mpl.rc('figure', titlesize=100)

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 22}

mpl.rc('font', **font)
plt.ion()

def norm(A):
    return (A-np.min(A))/(np.max(A)-np.min(A))

def myplot(*args, ax=None, title='', xlabel='Accelerating Voltage (V)', \
        ylabel='', xlim=[0,80], ylim=[0,1], xdv=5, legend=[]):
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
        ax.set_xticks(np.arange(xlim[0],xlim[1]+xdv, xdv))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    return k


fig, axs = plt.subplots(1, 1, figsize=(10,6))
# plt.subplots_adjust(left=0.05,right=0.98,bottom=0.05,top=0.95)


data = pd.read_csv('data/good/NewFile10.csv').to_numpy()[1:,:].astype(float)
data = data[data[:,0].argsort()][200:, :]

s, k = 0, 0
vi, vo = np.array([]), np.array([])
for i in range(1, len(data)):
    s+=data[i,1]
    k+=1
    if data[i, 0] != data[i-1, 0]:
        vo = np.append(vo, s/k)
        vi = np.append(vi, data[i, 0]*10)
        s,k=0,0

vo = norm(vo)
v2 = np.linspace(min(vi), max(vi), 1000)

spl = UnivariateSpline(vi, vo, k=4, s=0.01)

p0 = myplot(vi, vo, "o", v2, spl(v2), ax=axs, legend=["Data", "Quadratic Spline"], title="Franck-Hertz Response with Spline")

dspl = spl.derivative()
ddspl = dspl.derivative()
roots = np.array([r for r in dspl.roots() if ddspl(r) > 0.01 and ddspl(r)<10 and r > 5 and r < 80])

print(roots)
print(np.diff(roots))
for r in roots:
    plt.axvline(x=r, linestyle='--', color='r', alpha=0.3)


plt.savefig("oneplot10.png")

input("Press Enter to continue...")
