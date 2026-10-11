import numpy as np
#from sklearn.preprocessing import scale
from scipy.ndimage import gaussian_filter
from numpy.fft import fft, ifft
from scipy.interpolate import interp1d
import dcor
from tqdm import tqdm
from scipy.signal import convolve,find_peaks
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.spatial import distance_matrix
from functions.utilities import *


def get_mask_from_indices(indices,n_neurons):
    mask=np.zeros(n_neurons)
    mask[indices]=1
    mask=mask.astype(bool)
    return mask

def get_indices_with_higher_metric(indices_metrics_list):
    indices_higher_list=[]

    for i in range(len(indices_metrics_list)):

        indices,metric=indices_metrics_list[i]

        other_indices_metrics_list=indices_metrics_list.copy()
        other_indices_metrics_list.pop(i)

        indices_higher=[]
        for index in indices:
            max_flag=True
            for other_indices,other_metric in other_indices_metrics_list: 
                if index in other_indices and metric[index]<other_metric[index]:
                    max_flag=False
            if max_flag:
                indices_higher.append(index)

        indices_higher_list.append(indices_higher)
    
    return indices_higher_list


def get_neurons_same_line(neuron,layers,coords,direction,max_dist=5):
    x_dist=np.abs(coords[0,neuron]-coords[0])
    y_dist=np.abs(coords[1,neuron]-coords[1])
    
    if direction=='x':
        neurons_same_line=np.nonzero((layers==layers[neuron])&(y_dist<max_dist))[0]
    elif direction=='y':
        neurons_same_line=np.nonzero((layers==layers[neuron])&(x_dist<max_dist))[0]
    elif direction=='z':
        neurons_same_line=np.nonzero((layers!=layers[neuron])&(np.sqrt(y_dist**2+x_dist**2)<max_dist))[0]
    else:
        print('not a valid direction')
        return
        
    neurons_same_line=neurons_same_line[neurons_same_line!=neuron]
    
    if direction=='x':
        distances_along_line=x_dist[neurons_same_line]
    elif direction=='y':
        distances_along_line=y_dist[neurons_same_line]
    elif direction=='z':
        z_dist=np.abs(coords[2,neuron]-coords[2])
        distances_along_line=z_dist[neurons_same_line]
        
    return neurons_same_line,distances_along_line


def compute_correlations_and_distances_same_line(dff,layers,coords,direction,n_samples=1000,max_dist=5,seed=None):

    if seed is not None:
        np.random.seed(seed)

    corr_same_line=[]
    dist_same_line=[]
    for i in tqdm(range(n_samples)):
        neuron=np.random.randint(len(dff))
        neurons_same_line,distances_along_line=get_neurons_same_line(neuron,layers,coords,direction,max_dist=max_dist)

        if len(neurons_same_line)>0:
            corr_same_line.append(np.corrcoef(dff[neuron],dff[neurons_same_line])[0,1:])
            dist_same_line.append(distances_along_line)
        
    corr_same_line=np.concatenate(corr_same_line)
    dist_same_line=np.concatenate(dist_same_line)

    return corr_same_line,dist_same_line



def calculate_correlations_mean_std_over_distance(corr,dist,n_bins=10,bin_width=10,first_bin_bound=0):
    bin_bounds=first_bin_bound+np.arange(n_bins+1)*bin_width#np.linspace(0,max_distance,n_distances)
    bin_centers=(bin_bounds[1:]+bin_bounds[:-1])/2
    corr_mean=np.zeros_like(bin_centers)
    corr_std=np.zeros_like(bin_centers)
    for i in range(len(bin_centers)):
        corr_mean[i]=np.mean(corr[(dist>=bin_bounds[i])&(dist<bin_bounds[i+1])])
        corr_std[i]=np.std(corr[(dist>=bin_bounds[i])&(dist<bin_bounds[i+1])])
    return corr_mean,corr_std,bin_centers


def calculate_correlations_all_neurons(dff,fraction_of_neurons=0.1,seed=None):
    if seed is not None:
        np.random.seed(seed)
    random_indices=np.random.choice(np.arange(len(dff)),round(fraction_of_neurons*len(dff)),replace=False)
    corrmat=np.corrcoef(dff[random_indices])
    iu=np.triu_indices(len(corrmat),1)
    return corrmat[iu]



def calculate_correlations_neurons_from_different_layers(dff,layers,all_pairs=True,fraction_of_neurons=0.1,
                                                                          n_samples=1000,n_neurons_per_sample=100,
                                                                          seed=None):
    '''
    if all_pairs=True then it calculates the correlations between neurons of different layers for all possible pairs of layers,
    it select a fraction fraction_of_neurons of neurons per layer.
    if all_pairs=False it selects n_samples pairs of layers at random, with probability proportional to the number of neurons,
    and it calculates the correlations between n_neurons_per_sample per layer.
    '''
    
    if seed is not None:
        np.random.seed(seed)
        
    layers_unique,n_neurons_per_layer=np.unique(layers,return_counts=True)

    if all_pairs:
        layer_pairs=[(layer1,layer2) for i,layer1 in enumerate(layers_unique) for layer2 in layers_unique[i+1:]]
        corr_out=[]
        for i in tqdm(range(len(layer_pairs))):
            indices_layer1=np.nonzero(layers==layer_pairs[i][0])[0]
            indices_layer2=np.nonzero(layers==layer_pairs[i][1])[0]
            random_indices1=np.random.choice(indices_layer1,round(fraction_of_neurons*len(indices_layer1)),replace=False)
            random_indices2=np.random.choice(indices_layer2,round(fraction_of_neurons*len(indices_layer2)),replace=False)
            corrmat=np.corrcoef(dff[random_indices1],dff[random_indices2])
            corr_out.append(corrmat[:len(random_indices1),len(random_indices1):].ravel())
        corr_out=np.concatenate(corr_out)
    

    else:
        fraction_per_layer=np.array(n_neurons_per_layer)/np.sum(n_neurons_per_layer)
        corr_out=[]
        for i in tqdm(range(n_samples)):
            random_layers=np.random.choice(layers_unique,2,replace=False,p=fraction_per_layer)
            indices_layer1=np.nonzero(layers==random_layers[0])[0]
            indices_layer2=np.nonzero(layers==random_layers[1])[0]
            random_indices1=np.random.choice(indices_layer1,min(len(indices_layer1),n_neurons_per_sample),replace=False)
            random_indices2=np.random.choice(indices_layer2,min(len(indices_layer2),n_neurons_per_sample),replace=False)
            corrmat=np.corrcoef(dff[random_indices1],dff[random_indices2])
            corr_out.append(corrmat[:len(random_indices1),len(random_indices1):].ravel())
        corr_out=np.concatenate(corr_out)

    return corr_out

def calculate_correlations_neurons_same_layers(dff,layers,all_layers=True,fraction_of_neurons=0.1,
                                                                          n_samples=1000,n_neurons_per_sample=100,
                                                                          seed=None):
    '''
    if all_layers=True then it calculates the correlations between neurons of the same layer for all possible layers,
    it select a fraction fraction_of_neurons of neurons per layer.
    if all_layers=False it selects n_samples layers at random, with probability proportional to the number of neurons,
    and it calculates the correlations between n_neurons_per_sample.
    '''
    
    if seed is not None:
        np.random.seed(seed)
        
    layers_unique,n_neurons_per_layer=np.unique(layers,return_counts=True)

    if all_layers:
        corr_in=[]
        for i in tqdm(range(len(layers_unique))):
            indices_layer=np.nonzero(layers==layers_unique[i])[0]
            random_indices=np.random.choice(indices_layer,round(fraction_of_neurons*len(indices_layer)),replace=False)
            corrmat=np.corrcoef(dff[random_indices])
            iu=np.triu_indices(len(corrmat),1)
            corr_in.append(corrmat[iu])
        corr_in=np.concatenate(corr_in)
    

    else:
        fraction_per_layer=np.array(n_neurons_per_layer)/np.sum(n_neurons_per_layer)

        corr_in=[]
        for i in tqdm(range(n_samples)):
            random_layer=np.random.choice(layers_unique,p=fraction_per_layer)
            indices_layer=np.nonzero(layers==random_layer)[0]
            random_indices=np.random.choice(indices_layer,min(len(indices_layer),n_neurons_per_sample),replace=False)
            corrmat=np.corrcoef(dff[random_indices])
            iu=np.triu_indices(len(corrmat),1)
            corr_in.append(corrmat[iu])
        corr_in=np.concatenate(corr_in)

    return corr_in


def calculate_correlations_and_distances_same_layers(dff,layers,coords,fraction_of_neurons=0.1,seed=None):
    '''
    calculate correlation and distance for neurons in the same layers
    '''
    
    if seed is not None:
        np.random.seed(seed)
        
    layers_unique,n_neurons_per_layer=np.unique(layers,return_counts=True)

    corr_same=[]
    dist_same=[]
    for i in tqdm(range(len(layers_unique))):
        indices_layer=np.nonzero(layers==layers_unique[i])[0]
        random_indices=np.random.choice(indices_layer,round(fraction_of_neurons*len(indices_layer)),replace=False)
        corrmat=np.corrcoef(dff[random_indices])
        dist=distance_matrix(coords.T[random_indices],coords.T[random_indices])
        iu=np.triu_indices(len(corrmat),1)
        corr_same.append(corrmat[iu])
        dist_same.append(dist[iu])
    corr_same=np.concatenate(corr_same)
    dist_same=np.concatenate(dist_same)

    return corr_same,dist_same



def get_taus_from_dff_trace(dff_trace,brain_times,prominence=0.25):
    peaks,_=find_peaks(dff_trace,prominence=prominence)      
    taus=[]
    heights=[]
    for peak in peaks:
        height_peak=dff_trace[peak]
        if height_peak<=0:
            continue
        time_peak=brain_times[peak]
        height_threshold=height_peak/np.e
        indices_subthr=np.where((brain_times>brain_times[peak])&(dff_trace<height_threshold))[0]
        if len(indices_subthr)==0:
            continue
        i_subthr=indices_subthr[0]
        time_at_threshold=brain_times[i_subthr-1]+(height_threshold-dff_trace[i_subthr-1])*(brain_times[i_subthr]-brain_times[i_subthr-1])/(dff_trace[i_subthr]-dff_trace[i_subthr-1])
        tau_est=time_at_threshold-time_peak
        if np.all(np.diff(dff_trace[peak:i_subthr+1])<0):
            taus.append(tau_est)
            heights.append(height_peak)
    return taus,heights

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def calculate_rolling_quantile(brain_times,fluorescence,q=0.1,window_width=2*60,remove_central_value=True):
    dt=np.mean(np.diff(brain_times))
    w=(int(window_width/dt)//2)*2+1 #w needs to be odd
    rolling_quantile=np.zeros_like(fluorescence)
    if len(fluorescence.shape)>1:
        for i in range(len(fluorescence)):
            rolling_chunks=rolling_window(np.pad(fluorescence[i],w//2,mode='symmetric'),w)
            if remove_central_value:
                rolling_chunks=np.delete(rolling_chunks,w//2,-1)
            rolling_quantile[i]=np.quantile(rolling_chunks,q,axis=-1)
    else:
        rolling_chunks=rolling_window(np.pad(fluorescence,w//2,mode='symmetric'),w)
        if remove_central_value:
            rolling_chunks=np.delete(rolling_chunks,w//2,-1)
        rolling_quantile=np.quantile(rolling_chunks,q,axis=-1)

    return rolling_quantile

def calculate_dff_baseline_corrected(brain_times,fluorescence,q=0.1,window_width=2*60,background=100,remove_central_value=True):
    rolling_quantile=calculate_rolling_quantile(brain_times,fluorescence,q=q,
                                                window_width=window_width,remove_central_value=remove_central_value)
    dff_q=np.zeros_like(fluorescence)
    if len(fluorescence.shape)>1:
        for i in range(len(fluorescence)):
            tmp=rolling_quantile[i]-background
            tmp[tmp<1]=1
            dff_q[i]=(fluorescence[i]-rolling_quantile[i])/tmp
    else:
        tmp=rolling_quantile-background
        tmp[tmp<1]=1
        dff_q=(fluorescence-rolling_quantile)/tmp
    return dff_q
    
def gaussian_filtering(signal,signal_times,sigma=0.75):
    dt=np.mean(np.diff(signal_times))
    return gaussian_filter(signal,sigma/dt)

def deconvolution_exponential(signal,signal_times,tau=3,kernel_size=10):
    dt=np.mean(np.diff(signal_times))
    signal=np.pad(signal,1,mode='edge') #pad the signal to regularize edge behavior
    n_points=int(kernel_size*tau/dt) #kernel size in units of tau, tau in units of signal_times
    t=np.linspace(0,(n_points-1)*dt,n_points)
    kernel=np.exp(-t/tau)/tau #exponential kernel
    #kernel=np.hstack((kernel,np.zeros(len(signal)-len(kernel)))) #zero pad the kernel to same length
    #H=fft(kernel)
    #deconvolved=np.real(ifft(fft(signal)/H))
    H=fft(kernel,n=len(signal))*dt
    S=fft(signal,n=len(signal))*dt
    deconvolved=np.real(ifft(S/H))/dt
    return deconvolved[1:-1]

def convolution_exponential(signal,signal_times,tau=3,kernel_size=10):
    dt=np.mean(np.diff(signal_times)) #spacing between successive timepoints
    n_points=int(kernel_size*tau/dt)
    kernel_times=np.linspace(-n_points*dt,n_points*dt,2*n_points+1) #linearly spaced array from -n_points*dt to n_points*dt with spacing dt
    kernel=np.exp(-kernel_times/tau)/tau #define kernel as exponential decay
    kernel[kernel_times<0]=0 #set to zero for negative times
    signal_conv=convolve(signal,kernel,mode='same')*dt #perform the convolution
    return signal_conv

def interpolate(signal,signal_times,output_times,kind='linear'):
    signal_interp_f=interp1d(signal_times,signal,kind=kind,fill_value='extrapolate')
    signal_interp=signal_interp_f(output_times)
    return signal_interp

def calculate_distance_covariance(signal,dff):
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]
    
    dcovs=np.zeros(n_neurons)
    for i in range(n_neurons):
        dcovs[i]=dcor.distance_covariance(signal_notnan,dff[i,mask_notnan])
    return dcovs

def calculate_distance_correlation(signal,dff):
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]
    
    dcorrs=np.zeros(n_neurons)
    for i in range(n_neurons):
        dcorrs[i]=dcor.distance_correlation(signal_notnan,dff[i,mask_notnan])
    return dcorrs

def calculate_correlation(signal,dff):
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]
    
    corrs=np.zeros(n_neurons)
    for i in range(n_neurons):
        corrs[i]=pearsonr(signal_notnan,dff[i,mask_notnan])[0]
    return corrs

def calculate_covariance(signal,dff):
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]
    
    signal_centered=signal_notnan-np.mean(signal_notnan)
    dff_centered=dff[:,mask_notnan]-np.mean(dff[:,mask_notnan],axis=1)[:,np.newaxis]
    
    covs=np.zeros(n_neurons)
    for i in range(n_neurons):
        covs[i]=np.mean(signal_centered*dff_centered[i])
    return covs


def calculate_mutual_info(signal,dff,n_neighbors=3,discrete_features='auto',random_state=42):
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]
    X=signal_notnan.reshape(-1, 1)

    
    mis=np.zeros(n_neurons)
    for i in range(n_neurons):
        mis[i]=mutual_info_regression(X,dff[i,mask_notnan],n_neighbors=n_neighbors,discrete_features=discrete_features,random_state=random_state)
    return mis


def calculate_mutual_info_permutations(signal,dff,n_resamples=10,n_neighbors=3,discrete_features='auto',random_state=42):
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]
    X=signal_notnan.reshape(-1, 1)

    mis_cp=np.zeros((n_resamples,n_neurons)) 
    autocorr_cp=np.zeros((n_resamples,n_neurons)) 
    dff_scaled=scale(dff,axis=1)
    
    for n in tqdm(range(n_resamples)):
        dff_shifted=cyclic_permutation(dff,axis=1)
        autocorr_cp[n]=np.mean(scale(dff_shifted,axis=1)*dff_scaled,axis=1)
        for i in range(n_neurons):
            mis_cp[n,i]=mutual_info_regression(X,dff_shifted[i,mask_notnan],n_neighbors=n_neighbors,discrete_features=discrete_features,random_state=random_state)
    return mis_cp,autocorr_cp


def cyclic_permutation(signal,axis=0):
    l=len(signal)
    i=np.random.randint(l)
    shifted_signal=np.roll(signal,i,axis=axis)
    return shifted_signal

def calculate_distance_covariance_permutations(signal,dff,n_resamples=10,seed=42):
    np.random.seed(seed)
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]

    dcovs_cp=np.zeros((n_resamples,n_neurons)) 
    autocorr_cp=np.zeros((n_resamples,n_neurons)) 
    dff_scaled=scale(dff,axis=1)
    
    for n in tqdm(range(n_resamples)):
        dff_shifted=cyclic_permutation(dff,axis=1)
        autocorr_cp[n]=np.mean(scale(dff_shifted,axis=1)*dff_scaled,axis=1)
        for i in range(n_neurons):
            dcovs_cp[n,i]=dcor.distance_covariance(signal_notnan,dff_shifted[i,mask_notnan])
    return dcovs_cp,autocorr_cp

def calculate_distance_correlation_permutations(signal,dff,n_resamples=10,seed=42):
    np.random.seed(seed)
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]

    dcorrs_cp=np.zeros((n_resamples,n_neurons)) 
    autocorr_cp=np.zeros((n_resamples,n_neurons)) 
    dff_scaled=scale(dff,axis=1)
    
    for n in tqdm(range(n_resamples)):
        dff_shifted=cyclic_permutation(dff,axis=1)
        autocorr_cp[n]=np.mean(scale(dff_shifted,axis=1)*dff_scaled,axis=1)
        for i in range(n_neurons):
            dcorrs_cp[n,i]=dcor.distance_correlation(signal_notnan,dff_shifted[i,mask_notnan])
    return dcorrs_cp,autocorr_cp


def calculate_correlation_permutations(signal,dff,n_resamples=10,seed=42):
    np.random.seed(seed)
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]
    
    corrs_cp=np.zeros((n_resamples,n_neurons)) 
    autocorr_cp=np.zeros((n_resamples,n_neurons)) 
    dff_scaled=scale(dff,axis=1)
    
    for n in tqdm(range(n_resamples)):
        dff_shifted=cyclic_permutation(dff,axis=1)
        autocorr_cp[n]=np.mean(scale(dff_shifted,axis=1)*dff_scaled,axis=1)
        for i in range(n_neurons):
            corrs_cp[n,i]=pearsonr(signal_notnan,dff_shifted[i,mask_notnan])[0]
    return corrs_cp,autocorr_cp
    

def calculate_covariance_permutations(signal,dff,n_resamples=10):
    n_neurons=len(dff)
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]
    
    signal_centered=signal_notnan-np.mean(signal_notnan)
    dff_centered=dff[:,mask_notnan]-np.mean(dff[:,mask_notnan],axis=1)[:,np.newaxis]
    
    covs_cp=np.zeros((n_resamples,n_neurons)) 
    autocorr_cp=np.zeros((n_resamples,n_neurons)) 
    dff_scaled=scale(dff,axis=1)
    
    for n in tqdm(range(n_resamples)):
        dff_shifted=cyclic_permutation(dff_centered,axis=1)
        autocorr_cp[n]=np.mean(scale(dff_shifted,axis=1)*dff_scaled,axis=1)
        for i in range(n_neurons):
            covs_cp[n,i]=np.mean(signal_centered*dff_shifted[i])
    return covs_cp,autocorr_cp


def calculate_pvalue(statistic,statistic_null):
    statistic_null=statistic_null.ravel() #flatten array in case it's not already one-dimensional
    statistic_null_sorted=np.sort(statistic_null) #sort the values in increasing order
    statistic_null_sorted=np.append(statistic_null_sorted,statistic_null_sorted[-1])
    probability_of_smaller_values=np.arange(len(statistic_null_sorted))/(len(statistic_null_sorted)-1) #express the probability of finding a smaller value of the statistic
    empirical_distribution_function=interp1d(statistic_null_sorted,probability_of_smaller_values,kind='next',fill_value=(0,1),bounds_error=False) #interpolate to get the eCDF
    pvalue=np.zeros(len(statistic))
    for i in range(len(statistic)):
        pvalue[i]=1-empirical_distribution_function(statistic[i]) #probability of finding a larger value according to the null distribution
    return pvalue


def find_significant_neurons(statistic,statistic_null,alpha_bh=0.01):
    pvalue=calculate_pvalue(statistic,statistic_null)
    
    n_neurons=len(pvalue)
    indices=np.arange(n_neurons)+1
    sorted_pvalues=np.sort(pvalue)
    line_bh=indices/n_neurons*alpha_bh

    indices_inequality=np.where(sorted_pvalues<=line_bh)[0] #find all indices for which the p-value is below the line
    if len(indices_inequality)==0:
        n_significant_neurons=0
    else:
        n_significant_neurons=np.max(indices_inequality)+1 #largest index

    indices_sorted=np.argsort(-statistic) #sort indices with decreasing value of the statistic
    indices_thr=indices_sorted[:n_significant_neurons] #only take the ones for which the statistic is significantly larger than zero
    mask=np.zeros(n_neurons) #create a mask for the neurons for which we reject the null hypothesis
    mask[indices_thr]=1
    mask=mask.astype(bool)
    return indices_thr,mask


def calculate_tuning_curve(signal,dff,indices,n_bins=10,bin_edges=None,discrete=False,median=False,weights=None,together=False):
    
    mask_notnan=~np.isnan(signal)
    signal_notnan=signal[mask_notnan]
    
    if discrete:
        signal_values=np.unique(signal_notnan)
    else:
        if bin_edges is None:
            bin_edges=np.histogram_bin_edges(signal_notnan,n_bins)
        signal_values=(bin_edges[1:]+bin_edges[:-1])/2
        
    dff_mean=np.zeros(len(signal_values))
    dff_std=np.zeros(len(signal_values))

    for i in range(len(signal_values)):
        if discrete:
            mask_i=signal_notnan==signal_values[i]
        else:
            mask_i=(signal_notnan>bin_edges[i])&(signal_notnan<=bin_edges[i+1])
        
        if together:
            
            dff_mean_per_neuron=np.mean(dff[indices][:,mask_notnan][:,mask_i],axis=-1)
            dff_std_per_neuron=np.std(dff[indices][:,mask_notnan][:,mask_i],axis=-1)
            
            dff_mean[i]=np.median(dff_mean_per_neuron)
            dff_std[i]=(np.median(dff_mean_per_neuron+dff_std_per_neuron)-np.median(dff_mean_per_neuron-dff_std_per_neuron))/2

            
            #weights=1/dff_std_per_neuron**2
            #dff_mean[i]=np.sum(weights*dff_mean_per_neuron)/np.sum(weights)
            #dff_mean[i]=np.median(dff_mean_per_neuron)
            #dff_std[i]=1/np.sqrt(np.sum(weights))
            #dff_std[i]=np.median(dff_std_per_neuron)/np.sqrt(len(indices))
            
            
            #dff_mean[i]=np.median(dff[indices][:,mask_notnan][:,mask_i])
            #dff_std[i]=median_abs_deviation(dff[indices][:,mask_notnan][:,mask_i],axis=None)

            #dff_mean_per_neuron_per_time=np.mean(dff[indices][:,mask_notnan][:,mask_i],axis=0)
            #dff_std_per_neuron_per_time=np.std(dff[indices][:,mask_notnan][:,mask_i],axis=0)
            
            #dff_mean[i]=np.mean(dff_mean_per_neuron_per_time)
            #dff_std[i]=np.std(dff_mean_per_neuron_per_time)
            #dff_mean[i]=np.median(dff_mean_per_neuron_per_time)
            #dff_std[i]=median_abs_deviation(dff_mean_per_neuron_per_time,axis=None)

        else:
            
            if median:
                dff_mean_per_neuron=np.mean(dff[indices][:,mask_notnan][:,mask_i],axis=-1)
                dff_mean[i]=np.median(dff_mean_per_neuron)
                dff_std[i]=median_abs_deviation(dff_mean_per_neuron,axis=None)
            else:
                dff_mean_per_neuron=np.mean(dff[indices][:,mask_notnan][:,mask_i],axis=-1)
                if weights is None:
                    dff_mean[i]=np.mean(dff_mean_per_neuron)
                    dff_std[i]=np.std(dff_mean_per_neuron)
                else:
                    dff_mean[i]=np.sum(weights*dff_mean_per_neuron)/np.sum(weights)
                    dff_std[i]=np.sqrt(np.sum(weights*(dff_mean_per_neuron-dff_mean[i])**2))/np.sum(weights)
        

    return signal_values,dff_mean,dff_std
    
    
    
def calculate_r2_permutations(regressors,dff,n_resamples=10):
    n_neurons=len(dff)

    r2_cp=np.zeros((n_resamples,n_neurons)) 
    autocorr_cp=np.zeros((n_resamples,n_neurons)) 
    dff_scaled=scale(dff,axis=1)
    
    for n in tqdm(range(n_resamples)):
        dff_shifted=cyclic_permutation(dff,axis=1)
        autocorr_cp[n]=np.mean(scale(dff_shifted,axis=1)*dff_scaled,axis=1)
        reg=LinearRegression().fit(regressors.T,dff_shifted.T)
        r2_cp[n]=r2_score(dff_shifted.T,reg.predict(regressors.T),multioutput='raw_values')
    
    return r2_cp,autocorr_cp





