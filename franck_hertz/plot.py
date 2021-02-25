import os
import numpy as np
import pandas as pd
import argparse

# Make plots readable
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('lines', markersize=2)
plt.ion()


mypath = 'data/good/'
files = [f for f in os.listdir(mypath) if (os.path.isfile(os.path.join(mypath, f)) and '.csv' in f)]

for f in range(len(files)):
    print(files[f])
    file_name = os.path.join(mypath, files[f])
    data = pd.read_csv(file_name).to_numpy()

    data = data[1:,:].astype(float)
    
    #r, c = int(np.floor(f/n)), f % n

    # Visualize Data
    t, d = data[:,0]*10, data[:,1]
    #ax=axs[r,c]
    plt.figure(f)
    plt.scatter(t,d)
    plt.xlabel('Accelerating Voltage (V)')
    plt.ylabel('Measured Voltage (Arbitary)')
    plt.title(files[f])
    #plt.xlim([0, 8])
    plt.grid(True)
    plt.show()




input("Press Enter to continue...")
