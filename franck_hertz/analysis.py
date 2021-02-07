import os
import numpy as np
import pandas as pd

# Make plots readable
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('lines', markersize=2)
plt.ion()


# Read Data
#file_name = '/run/media/psimmerl/3501_data/NewFile8.csv' #'/home/psimmerl/Downloads/break_179158.csv'

mypath = 'data/'
files = [f for f in os.listdir(mypath) if (os.path.isfile(os.path.join(mypath, f)) and '.csv' in f)]
#n = int(np.ceil(np.sqrt(len(files))))
#fig, axs = plt.subplots(n,n)

for f in range(len(files)):
    print(files[f])
    file_name = os.path.join(mypath, files[f])
    data = pd.read_csv(file_name).to_numpy()

    data = data[1:,:].astype(np.float)
    
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

# d_fft = np.fft.fft(d)
# freq = np.fft.fftfreq(t.shape[-1])

# plt.figure(2)
# plt.plot(freq, d_fft.real, freq, d_fft.imag)
# plt.grid(True)
# plt.show()


# #Fit Sawtooth
# from scipy import signal
# plt.figure(3)
# t = np.linspace(0, 1, 500)
# plt.plot(t, (t+1)*(signal.sawtooth(2 * np.pi * 5 * t)+t+1))
# plt.grid(True)
# plt.show()

# # Ei = gaussian?
# # Ef = Ei - Sum[p(collision)*Ec]



input("Press Enter to continue...")
