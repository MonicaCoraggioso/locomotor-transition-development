import os
import numpy as np
import h5py
import shutil
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation,iqr
from matplotlib.colors import LogNorm
from matplotlib.colors import to_rgb
from datetime import date

textwidth=7
textwidth1=2.24
textwidth2=4.76
textwidth3=7.24
textwidthSI=6.5
textwidththesis=5.9


distance=0.5
strength_factor=0.2
turning_strength_factor=200

def latex(fontsize=8,serif=True):
    plt.close('all')
    plt.rcdefaults()
    plt.rcParams['text.usetex']= True
    if serif:
        plt.rcParams['font.family']= 'serif'
    plt.rcParams['font.size']=fontsize

def default():
    plt.close('all')
    plt.rcdefaults()
    
def arial(fontsize=8):
    plt.close('all')
    plt.rcdefaults()
    plt.rcParams['font.family']= 'Arial'
    plt.rcParams['font.size']=fontsize


def round_sig(x, sig):
    x = np.array(x, dtype=np.float64)
    mags = np.power(10, sig - np.floor(np.log10(np.abs(x))) - 1)
    rounded = np.round(x * mags) / mags
    return rounded


def count_consecutive_ones(arr,start_index):
    count=0
    for i in range(start_index,len(arr)):
        if arr[i]==1:
            count+=1
        else:
            break
    return count


def count_until_zero(arr,start_index):
    count=0
    for i in range(start_index,len(arr)):
        if arr[i]!=0:
            count+=1
        else:
            break
    return count


    
    
def extend_zeros(mask, k):
    extended_mask = mask.copy()
    zeros_indices = np.where(mask == 0)[0]

    for zero_index in zeros_indices:
        start_index = max(0, zero_index - k)
        end_index = min(len(mask), zero_index + k + 1)
        extended_mask[start_index:end_index] = 0

    return extended_mask


def set_short_sequences_to_zero(mask, min_length):
    extended_mask = mask.copy()
    current_sequence_length = 0

    for i in range(len(mask)):
        if mask[i] == 1:
            current_sequence_length += 1
        else:
            if current_sequence_length < min_length:
                extended_mask[i - current_sequence_length:i] = 0
            current_sequence_length = 0

    if current_sequence_length < min_length and current_sequence_length > 0:
        extended_mask[-current_sequence_length:] = 0

    return extended_mask



def find_consecutive_sequences(array, condition):
    sequences = []
    current_sequence = []
    
    for num in array:
        if condition(num):
            current_sequence.append(num)
        else:
            if current_sequence:
                sequences.append(current_sequence)
            current_sequence = []
    
    if current_sequence:
        sequences.append(current_sequence)
    
    return sequences

def find_consecutive_sequences_mask(array, mask):
    sequences = []
    current_sequence = []
    
    for i,num in enumerate(array):
        if mask[i]:
            current_sequence.append(num)
        else:
            if current_sequence:
                sequences.append(current_sequence)
            current_sequence = []
    
    if current_sequence:
        sequences.append(current_sequence)
    
    return sequences

    
def asymmetric_sigmoid(x,ap=1,am=1,k=1):
    if x==0:
        return 0
    else:
        a=np.heaviside(x,0)*ap+np.heaviside(-x,0)*am
        return a*np.tanh(k*x/a)

def weber_response_cont_sym(x,b,a):
    if np.isscalar(x):
        if x<0:
            return -a*np.log(1-b/a*x)
        else:
            return a*np.log(1+b/a*x)
    else:
        y=np.copy(x)  
        y[x>=0]=a*np.log(1+b/a*x[x>=0])
        y[x<0]=-a*np.log(1-b/a*x[x<0])
        
        return y

def weber_response_asymmetric(x,x0,k1,k2,b=0,xmax=np.inf):
    if np.isscalar(x):
        if x<-xmax:
            return -k2*np.log(xmax/x0)+b
        elif  x>=-xmax and x<-x0:
            return -k2*np.log(-x/x0)+b
        elif x>=-x0 and x<=x0:
            return b
        elif x>x0 and x<xmax:
            return k1*np.log(x/x0)+b
        else:
            return k1*np.log(xmax/x0)+b
    else:
        y=np.copy(x)    
        y[(x<=x0)&(x>=-x0)]=0
        y[(x>=x0)&(x<=xmax)]=k1*np.log(x[(x>=x0)&(x<=xmax)]/x0)
        y[x>xmax]=k1*np.log(xmax/x0)
        y[(x<=-x0)&(x>=-xmax)]=-k2*np.log(-x[(x<=-x0)&(x>=-xmax)]/x0)
        y[x<-xmax]=-k2*np.log(xmax/x0)

        return y+b

def weber_response(x,x0,xmax,k):
    if np.isscalar(x):
        if x<=-xmax:
            return -k*np.log(xmax/x0)
        elif x>-xmax and x<-x0:
            return -k*np.log(-x/x0)
        elif x>=-x0 and x<=x0:
            return 0
        elif x>x0 and x<xmax:
            return k*np.log(x/x0)
        elif x>=xmax:
            return k*np.log(xmax/x0)
    else:
        y=np.copy(x)
        y[x<=xmax]=-k*np.log(xmax/x0)
        y[(x>-xmax)&(x<-x0)]=-k*np.log(-x[(x>-xmax)&(x<-x0)]/x0)
        y[(x>=-x0)&(x<=x0)]=0
        y[(x>x0)&(x<xmax)]=k*np.log(x[(x>x0)&(x<xmax)]/x0)
        y[x>xmax]=k*np.log(xmax/x0)
        return y

def weber_response_no_max(x,x0,k):
    if np.isscalar(x):
        if  x<-x0:
            return -k*np.log(-x/x0)
        elif x>=-x0 and x<=x0:
            return 0
        elif x>x0 :
            return k*np.log(x/x0)
    else:
        y=np.copy(x)
        y[(x<-x0)]=-k*np.log(-x[(x<-x0)]/x0)
        y[(x>=-x0)&(x<=x0)]=0
        y[(x>x0)]=k*np.log(x[(x>x0)]/x0)
        return y
        
def weber_response_no_threshold(x,k,x0=1,r=1):
    if np.isscalar(x):
        if x<0:
            return -k/r*np.log(1-x/x0)
        else:
            return k*r*np.log(1+x/x0)
    else:
        y=np.copy(x)
        y[x<0]=-k/r*np.log(1-x[x<0]/x0)
        y[x>=0]=k*r*np.log(1+x[x>=0]/x0)
        return y


def find_quartiles(y):
    y_m = np.nanquantile(y,0.5) 
    y_iqp = np.nanquantile(y,0.75)-np.nanquantile(y,0.5) 
    y_iqm = np.nanquantile(y,0.5)-np.nanquantile(y,0.25) 

    return y_m,y_iqp,y_iqm

    

def find_fwhm(x,y):
    peak_idx = np.argmax(y) 
    peak_y = y[peak_idx] 
    half_max = peak_y / 2

    x_peak=x[peak_idx]
    
    left_indices = np.where(y[:peak_idx] <= half_max)[0]
    right_indices = np.where(y[peak_idx:] <= half_max)[0] + peak_idx 

    if left_indices.size > 0:
        left_idx = left_indices[-1]
        x_left=x[left_idx]+(x[left_idx+1]-x[left_idx])*(half_max-y[left_idx])/(y[left_idx+1]-y[left_idx])
    else:
        x_left = x[0] 

    if right_indices.size > 0:
        right_idx = right_indices[0] 
        x_right=x[right_idx]+(x[right_idx-1]-x[right_idx])*(half_max-y[right_idx])/(y[right_idx-1]-y[right_idx])
    else:
        x_right = x[-1]

    return x_peak,x_left,x_right  

def calculate_crosscorr(x,y,dt):
    '''
    calculates cross-correlation between x and y, it also returns the corresponding time shifts, 
    positive values correspond to a forward shift of y with respect to x
    '''
    N=max(len(x),len(y))
    dis=np.arange(len(x)+len(y)-1)-(N-1)
    dts=dis*dt
    return np.correlate(x,y,mode='full')/N,dts

def get_logarithmic_boundaries(xs):
    dlogx=(np.log(xs)[1]-np.log(xs)[0])/2
    xbs=np.exp(np.concatenate(([np.log(xs)[0]-dlogx],np.log(xs)+dlogx)))
    return xbs

def get_linear_boundaries(xs):
    dx=(xs[1]-xs[0])/2
    xbs=np.concatenate(([xs[0]-dx],xs+dx))
    return xbs



def copy_pickles_previous_run(path,today,run):
    if run>1:
        path_pickles=os.path.join(path,today,f'run_{run:02d}','pickles')
        path_pickles_previous_run=os.path.join(path,today,f'run_{run-1:02d}','pickles')
        files=os.listdir(path_pickles_previous_run)
        for fname in files:
            shutil.copy2(os.path.join(path_pickles_previous_run,fname), path_pickles)
        print('Copied files')
    else:
        print('Run must be larger than 1')

def scale(X,axis=0,with_mean=True,with_std=True):
    X = np.copy(X)
    if with_mean:
        mean_ = np.nanmean(X, axis)
    if with_std:
        scale_ = np.nanstd(X, axis)
    Xr = np.rollaxis(X, axis)
    if with_mean:
        Xr -= mean_
    if with_std:
        Xr /= scale_
    return X


    mean=np.nanmean(signal,axis=axis)
    std=np.nanmean(signal,axis=axis)
    if with_mean:
        signal-=mean
    if with_std:
        signal/=std
    return signal

from scipy.signal import correlate

def autocorrelation(signal):
    n = len(signal)
    mean = np.mean(signal)
    centered_signal = signal - mean
    autocorr = correlate(centered_signal, centered_signal, mode='full') / (n * np.var(signal))
    return autocorr[len(signal)-1:]


def load_fpga_params(file_format,path,date,run):
    '''
    get parameters of fpga either from h5 or txt file 
    '''
    runstr=f'run_{run:02d}'
    path_run=os.path.join(path,date,runstr)
    assert os.path.exists(path_run), 'path does not exist'

    if file_format=='h5':
        
        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            if f['exp_ps/trigger_fpga'][...]:
                return save_group_as_dict(f,'fpga_ps')

    elif file_format=='txt':
        
        if not os.path.exists(os.path.join(path_run,'parameters.txt')):
            print('no parameter file found')
            return

        with open(os.path.join(path_run,'parameters.txt'),'r') as f:
            
            fpga_ps={}
            lines = f.readlines()

            for line in lines:
                if line.split()[1]=='(volt)' or line.split()[1]=='(um)' or line.split()[1]=='(ms)' or line.split()[1]=='(s)' or line.split()[1]=='(um/ms)':
                    fpga_ps[line.split()[0]]=float(line.split()[-1])
                elif line.split()[1]=='(int)':
                    fpga_ps[line.split()[0]]=int(line.split()[-1])
                else:
                    fpga_ps[line.split()[0]]=line.split()[-1]
                
            return fpga_ps



def get_path_run(path,date,run):
    runstr=f'run_{run:02d}'
    path_run=os.path.join(path,date,runstr)
    return path_run

def raise_to_power(signal,exponent):
    '''
    raise an array to a non integer power by first taking the absolute values and then remultiplying by the signs
    '''
    return np.sign(signal)*np.abs(signal)**exponent


def rotate_and_flip(frame,ps):
    '''
    flip and rotate an image by a multiple of 90 deg, it's fast
    '''
    if ps['transpose']:
        frame=frame.T
    if ps['flip_x']:
        frame=frame[::-1]
    if ps['flip_y']:
        frame=frame[:,::-1]
    return frame


def print_attrs(name, obj):
    '''
    print all attributes in a h5 file
    '''
    print(name)
    for key, val in obj.attrs.items():
        print(f'  {key}:{val}')

def save_group_as_dict(f,groupname):
    '''
    create a dictionary from a h5 dataset
    '''
    groupdict={}
    for key in list(f[groupname].keys()):
        if f[groupname][key][...].shape==():
            groupdict[key]=f[groupname][key][...].item()
        else:
            groupdict[key]=f[groupname][key][...]
    return groupdict


def get_filenames(path,date,run):
    '''
    get filenames of h5 files
    '''
    runstr=f'run_{run:02d}'
    path_h5=os.path.join(path,date,runstr)
    assert os.path.exists(path_h5), f'{path_h5} does not exist'
    filenames={'visual_flow':os.path.join(path_h5,'visual_flow_'+date+'_'+runstr+'.h5'),
               'tail_pictures':os.path.join(path_h5,'tail_pictures_'+date+'_'+runstr+'.h5'),
               'stimulus_pictures':os.path.join(path_h5,'stimulus_pictures_'+date+'_'+runstr+'.h5'),
               'brain_data':os.path.join(path_h5,'brain_data_'+date+'_'+runstr+'.h5'),
               'brain_analysis':os.path.join(path_h5,'brain_analysis_'+date+'_'+runstr+'.h5'),
               'stack':os.path.join(path_h5,'stack_'+date+'_'+runstr+'.mha')}
    return filenames

def five_point_derivative(signal,times):
    '''
    calculate derivative using 5 points
    '''
    f=np.pad(signal,2,mode='edge')
    dt=np.mean(np.diff(times))
    derivative=(-f[4:]+8*f[3:-1]-8*f[1:-3]+f[:-4])/(12*dt)
    return derivative
    
    
def stationary_bootstrap(signal,average_block_size=240):
    """resample a time series using stationary bootstrap, it samples with replacement random blocks of sizes following a geometric distribution.

    Parameters :
    ------------
    signal : array
        original time series, the first dimension corresponds to time, shape (n_timepoints,...)
        
    average_block_size : integer
        average number of values in a block during resampling
        default : 240
        
    Return :
    --------
    resampled_signal : array
        bootstrapped time series
    """
    l=len(signal)
    p=1/average_block_size #probability of choosing a new random value 
    resampled_signal=np.zeros_like(signal) #initialize the output array
    j=np.random.randint(l) #index of the first random value 
    resampled_signal[0]=signal[j] #assign the first random value
    for i in range(1,l): #loop over the next positions in the new array
        if np.random.random()<1-p: #sample a random number to choose in between the two options
            j=(j+1)%l #take the next index
        else:
            j=np.random.randint(l) #take a random index
        resampled_signal[i]=signal[j] #assign the value to the new array
    
    return resampled_signal

def bootstrap(signal,l=None):
    if l is None:
        l=len(signal)
    random_indices=np.random.randint(l,size=l)
    resampled_signal=signal[random_indices]
    return resampled_signal


def rolling_mean(data, window_size, overlap_fraction):
    step_size = int(window_size * (1 - overlap_fraction))
    rolling_means = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        rolling_means.append(np.mean(window))
    return rolling_means

def asymmetry(x,r):
    if x>0:
        return r*x
    else:
        return 1/r*x
    
    
def logstar(omega,omegac):
    if np.isscalar(omega):
        if omega<0:
            return -np.log(1-omega/omegac)
        else:
            return np.log(1+omega/omegac)
    else:
        y=np.copy(omega)  
        y[omega>=0]=np.log(1+omega[omega>=0]/omegac)
        y[omega<0]=-np.log(1-omega[omega<0]/omegac)
        
        return y