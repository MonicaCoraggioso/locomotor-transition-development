import numpy as np
import h5py
from ipywidgets import interact
from scipy import fft
from scipy.signal import welch,periodogram,hilbert
from scipy.ndimage import gaussian_filter
from functions.utilities import *
from functions.freely_swimming import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import theilslopes,linregress
from tqdm import tqdm
from itertools import groupby

def get_colors():
    #Age colorpalette:
    color_y = '#D5E600'
    color_m = "#919826"
    color_o = '#32332B' 
    
    # eta(boutID) colorpalette:
    color_low_boutID = '#E44AC0'
    color_med_boutID = '#2341A3'
    color_high_boutID = '#6FEAA3'
    
     # vout-cont colorpalette:
    color_cont = '#DEBAEA'
    color_bout = '#A8D6B7'

    return color_y, color_m, color_o, color_low_boutID, color_med_boutID, color_high_boutID, color_cont, color_bout

def stars_from_p(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'  # not significant
    
def get_peak(ts,impulse_resp,tau=0):
    imax=np.argmax(np.cumsum(impulse_resp))
    kernel_times=ts[:imax+1]+tau
    kernel_points=impulse_resp[:imax+1]
    kernel_times=np.insert(kernel_times,0,[0,tau])
    kernel_points=np.insert(kernel_points,0,[0,0])
    return kernel_times,kernel_points


def calculate_fourier_spectrum(signal,times,tmin=None,tmax=None,fs=None):
    '''
    calculate fourier spectrum of a signal
    '''
    if tmin is None:
        tmin=times[0]
    if tmax is None:
        tmax=times[-1]   
    times_selected=tuple([(times>=tmin)&(times<=tmax)])
    if fs is None:
        fs=1/np.mean(np.diff(times[times_selected]))
    signal_selected=signal[times_selected]-np.mean(signal[times_selected])
    power=np.abs(fft.rfft(signal_selected)/fs)**2
    frequencies=fft.rfftfreq(len(signal_selected),1/fs)
    return frequencies,power

def calculate_fourier_power_spectral_density(signal,times,tmin=None,tmax=None,fs=None):
    '''
    calculate fourier spectrum of a signal
    '''
    if tmin is None:
        tmin=times[0]
    if tmax is None:
        tmax=times[-1]   
    times_selected=tuple([(times>=tmin)&(times<=tmax)])
    if fs is None:
        fs=1/np.mean(np.diff(times[times_selected]))
    signal_selected=signal[times_selected]-np.mean(signal[times_selected])
    power=2*np.abs(fft.rfft(signal_selected)/fs)**2/(len(signal_selected)/fs)
    frequencies=fft.rfftfreq(len(signal_selected),1/fs)
    return frequencies,power

def calculate_fourier_spectrum_periodogram(signal,times=None,tmin=None,tmax=None,fs=None,scaling='density'):
    '''
    calculate fourier spectrum of a signal using the periodogram algorithm
    '''
    if tmin is None:
        tmin=times[0]
    if tmax is None:
        tmax=times[-1]   
    times_selected=tuple([(times>=tmin)&(times<=tmax)])
    if fs is None:
        fs=1/np.mean(np.diff(times[times_selected]))
    signal_selected=signal[times_selected]-np.mean(signal[times_selected])
    
    frequencies,power=periodogram(signal_selected,fs,scaling=scaling)

    return frequencies,power


def calculate_fourier_spectrum_welch(signal,times=None,tmin=None,tmax=None,fs=None,window_length=None,scaling='density',return_onesided=True):
    '''
    calculate fourier spectrum of a signal using the welch algorithm
    '''
    if times is None:
        signal_selected=signal
    else:
        if tmin is None:
            tmin=times[0]
        if tmax is None:
            tmax=times[-1]   
        times_selected=tuple([(times>=tmin)&(times<=tmax)])
        signal_selected=signal[times_selected]

    if fs is None:
        fs=1/np.mean(np.diff(times[times_selected]))
            
    if window_length is None:
        nperseg=None
    else:
        nperseg=round(window_length*fs)
        
    frequencies,power=welch(signal_selected,fs,nperseg=nperseg,scaling=scaling,return_onesided=return_onesided)

    return frequencies,power


def calculate_acf(signal, times, max_t, dt = None):
    
    # Compute the autocorrelation function (acf) of a signal

    if dt is None:
        dt = times.diff()

    t = times
    X = signal
    
    max_index = int(max_t/dt)+1
    
    autocorr_vals = [pd.Series(X).autocorr(lag) for lag in range(max_index)] 
        
    return t[:max_index], autocorr_vals

def normalize_y(val):
    y_min = np.min(val)
    y_max = np.max(val)
    return (val - y_min) / (y_max - y_min)