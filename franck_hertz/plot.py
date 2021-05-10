import os
import numpy as np
import pandas as pd

# Make plots readable
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)
mpl.rc('lines', markersize=3)

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 22}

mpl.rc('font', **font)

plt.ion()



mypath = 'data/good/'
files = ['NewFile10.csv']
# files = [f for f in os.listdir(mypath) if (os.path.isfile(os.path.join(mypath, f)) and '.csv' in f)]

for f in range(len(files)):
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    print(files[f])
    file_name = os.path.join(mypath, files[f])
    data = pd.read_csv(file_name).to_numpy()

    data = data[1:,:].astype(float)
    
    #r, c = int(np.floor(f/n)), f % n

    # Visualize Data
    t, d = data[:,0]*10, data[:,1]
    #ax=axs[r,c]
    #ax.figure(f)
    ax.scatter(t,d)
    ax.set_xlabel('Accelerating Voltage (V)')
    ax.set_ylabel('Measured Voltage (Arbitary)')
    ax.set_title("Franck-Hertz Response (Fails Spline)")#files[f])#
    ax.set_xlim(0,80)
    ax.set_ylim(0,1)
    ax.set_xticks(np.arange(0,85, 5))
    #plt.xlim([0, 8])
    ax.grid(True)
    fig.tight_layout()

    # ax.show()

plt.savefig("oneplot10_raw.png")



input("Press Enter to continue...")
