import os
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy.integrate import quad, dblquad, nquad
from matplotlib.widgets import Slider, Button

# Make plots readable
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('lines', markersize=2)
plt.ion()

a,E,mu,tau,c,k,tol=1,5,2.5,1/5,1,-4,-1

def sawtooth(x, a=a, E=E):
    return a/E*x-np.floor(x/E)

def fermi(x, mu=mu, tau=tau):
    return 1/(np.exp((x-mu)/tau)+1)

def current(x, c=c, k=k):
    return c*x+k#c*np.exp(k*x)

def norm(A):
    return (A-np.min(A))/(np.max(A)-np.min(A))

vmin, vmax, dv = 0, 80, 0.1

v = np.arange(vmin, vmax+dv, dv)

fig, axs = plt.subplots(2,2,figsize=(10,10))
plt.subplots_adjust(left=0.05,right=0.98,bottom=0.17,top=0.95)

def myplot(*args, ax=None, legend=[], title="", xlim=[vmin,vmax],ylim=[0,1], xdv=E):
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
    return k


data = pd.read_csv('data/good/NewFile7.csv').to_numpy()
data = data[1:,:].astype(float)
t, d = data[:,0]*10, norm(data[:,1])

zs  = np.zeros(np.size(v))
p0 = myplot(v, zs, v, zs, v, zs, ax=axs[0,0], legend=["Fermi(mu,tau)","Exp(k)","Sawtooth(a,E)"], title="Inputs")
p1 = myplot(v, zs, v, zs, v, zs, ax=axs[0,1], legend=["Saw*Fermi", "Saw*Exp", "Fermi*Exp"], title="Convolutions")
gs = axs[1, 1].get_gridspec()
axs[1,0].remove()
axs[1,1].remove()
axbig = fig.add_subplot(gs[1,:])
p2 = myplot(v, zs, v, zs, t, d, "o", ax=axbig, legend=["Conv Integral","FFT", "data"],title="Total")

from multiprocessing import Pool
from functools import partial

def vsfi(v=v,a=a,E=E,mu=mu,tau=tau,c=c,k=k,tol=tol):
    return dblquad(lambda v1, v2, v=v,a=a,E=E,mu=mu,tau=tau,c=c,k=k: \
            sawtooth(v1,a,E)*fermi(v-v1-v2,mu,tau)*current(v2,c,k), \
            vmin, v, lambda x: vmin, lambda x,v=v: v, \
            epsabs=tol, epsrel=tol)[0]
def vsf(v=v,a=a,E=E,mu=mu,tau=tau,tol=tol):
    return quad(lambda v1, v=v,a=a,E=E,mu=mu,tau=tau: \
            sawtooth(v1, a, E)*fermi(v-v1, mu, tau), \
            vmin, v, epsabs=tol, epsrel=tol)[0]
def vsi(v=v,a=a,E=E,c=c,k=k,tol=tol):
    return quad(lambda v1, v=v,a=a,E=E,c=c,k=k: \
        sawtooth(v1, a, E)*current(v-v1, c, k), \
        vmin, v, epsabs=tol, epsrel=tol)[0]
def vfi(v=v,mu=mu,tau=tau,c=c,k=k,tol=tol):
    return quad(lambda v1, v=v,mu=mu,tau=tau,c=c,k=k: \
            fermi(v1, mu, tau)*current(v-v1, c, k), \
            vmin, v, epsabs=tol, epsrel=tol)[0]

def update(val):
    E,mu,tau,k=sE.val,smu.val,stau.val,10**(sk.val)
    c=sc.val#,sd.val,sa.val,a,
    a=1
    tol=10**(stol.val)#10e-4
    S, N, I = norm(sawtooth(v,a,E)), norm(fermi(v,mu,tau)), norm(current(v,c,k))
    
    # Sf, Nf, If = np.fft.rfft(S), np.fft.rfft(N), np.fft.rfft(I)

    # VSFI_fft = norm(np.fft.irfft(Sf*Nf*If, np.size(v)))
    # VSF_fft  = norm(np.fft.irfft(Sf*Nf, np.size(v)))
    # VSI_fft  = norm(np.fft.irfft(Sf*If, np.size(v)))
    # VFI_fft  = norm(np.fft.irfft(Nf*If, np.size(v)))

    # vsfi2 = partial(vsfi, a=a,E=E,mu=mu,tau=tau,c=c,k=k,tol=tol)
    # vsf2 = partial(vsf, a=a,E=E,mu=mu,tau=tau,tol=tol)
    # vsi2 = partial(vsi, a=a,E=E,c=c,k=k,tol=tol)
    # vfi2 = partial(vfi, mu=mu,tau=tau,c=c,k=k,tol=tol)
    
    # with Pool(8) as p:
    #     VSFI = p.map(vsfi2,v)
    #     VSF = p.map(vsf2,v)
    #     VSI = p.map(vsi2,v)
    #     VFI = p.map(vfi2,v)

    # VSFI, VSF, VSI, VFI = norm(VSFI), norm(VSF), norm(VSI), norm(VFI)
    # VSFI, VSF, VSI, VFI = VSFI_fft, VSF_fft, VSI_fft, VFI_fft
    VSFI = norm(signal.fftconvolve(S, signal.fftconvolve(N, I, 'same'), 'same'))
    VSF = norm(signal.fftconvolve(S, N, 'same'))
    VSI = norm(signal.fftconvolve(S, I, 'same'))
    VFI = norm(signal.fftconvolve(N, I, 'same'))   


    p0[0].set_ydata(N)
    p0[1].set_ydata(I)
    p0[2].set_ydata(S)
    p1[0].set_ydata(VSF)
    p1[1].set_ydata(VSI)
    p1[2].set_ydata(VFI)
    p2[0].set_ydata(VSFI)
    #p2[1].set_ydata(VSFI_fft)
    fig.canvas.draw_idle()

axtol  = plt.axes([0.07, 0.11, 0.85, 0.01])
#axa   = plt.axes([0.07, 0.09, 0.85, 0.01])
axE   = plt.axes([0.07, 0.07, 0.85, 0.01])
axmu  = plt.axes([0.07, 0.05, 0.85, 0.01])
axtau = plt.axes([0.07, 0.03, 0.85, 0.01])
axc   = plt.axes([0.07, 0.09, 0.85, 0.01])
axk   = plt.axes([0.07, 0.01, 0.85, 0.01])

stol = Slider(axtol, 'log(tol)', -8, 0, valinit=tol)
#sa   = Slider(axa, 'a', 0, 5, valinit=a)
sE   = Slider(axE, 'E', 0.1, 10, valinit=E)
smu  = Slider(axmu, 'mu', 0, 10, valinit=mu)
stau = Slider(axtau, 'tau', 0.001, 3, valinit=tau)
sc   = Slider(axc, 'c', 0, 10.0, valinit=c)
sk   = Slider(axk, 'log(k)', -8, 1.0, valinit=k)



resetax = plt.axes([0.9, 0.13, 0.05, 0.015])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    stol.reset()
    #sa.reset()
    sE.reset()
    smu.reset()
    stau.reset()
    sc.reset()
    sk.reset()

stol.on_changed(update)
#sa.on_changed(update)
sE.on_changed(update)
smu.on_changed(update)
stau.on_changed(update)
sc.on_changed(update)
sk.on_changed(update)

button.on_clicked(reset)

update(0) 

plt.show()
input("Press enter to continue...")

# p0 = myplot(v, zs, v, zs, v, zs, ax=axs[0,0], legend=["Fermi(mu,tau)", "Exp(k)", "Sawtooth(a,E)"], title="Inputs")
# p1 = myplot(v, zs, v, zs, ax=axs[0,1], legend=["ifft[fft[Saw]*fft[Fermi]]", "conv[Saw,Fermi]"], title="Saw*Fermi")
# p2 = myplot(v, zs, v, zs, ax=axs[1,0], legend=["ifft[fft[Saw]*fft[Exp]]", "conv[Saw,Exp]"], title="Saw*Exp")
# p3 = myplot(v, zs, v, zs, ax=axs[1,1], legend=["ifft[fft[Fermi]*fft[Exp]]", "conv[Fermi,Exp]"], title="Fermi*Exp")
# Sf, Nf, If = np.fft.rfft(S), np.fft.rfft(N), np.fft.rfft(I)

# VSFI_fft = norm(np.fft.irfft(Sf*Nf*If, np.size(v)))
# VSF_fft  = norm(np.fft.irfft(Sf*Nf, np.size(v)))
# VSI_fft  = norm(np.fft.irfft(Sf*If, np.size(v)))
# VFI_fft  = norm(np.fft.irfft(Nf*If, np.size(v)))

# p1[0].set_ydata(VSF_fft)    
# p2[0].set_ydata(VSI_fft)    
# p3[0].set_ydata(VFI_fft)

