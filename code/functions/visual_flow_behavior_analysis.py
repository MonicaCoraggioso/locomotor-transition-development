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

def calculate_slopes_per_fish_with_theilsen(params_list,deltavs_trials_list,strengthgradients_trials_list,seed=42,n_points=1000,min_quantile=0.1,max_quantile=0.9):
    np.random.seed(seed)
    params=params_list[0]
    slopes=np.full((len(params_list),len(params)),np.nan)
    slopesm=np.full((len(params_list),len(params)),np.nan)
    slopesp=np.full((len(params_list),len(params)),np.nan)
    for j in tqdm(range(len(params))):
        for i in range(len(params_list)):
            if len(deltavs_trials_list[i][j])>0:
                x=np.concatenate(deltavs_trials_list[i][j])
                y=np.concatenate(strengthgradients_trials_list[i][j])
                mask=(x>np.quantile(x,min_quantile))&(x<np.quantile(x,max_quantile))
                if np.any(mask):
                    x=x[mask]
                    y=y[mask]
                    random_indices=np.random.randint(len(x),size=n_points)
                    x=x[random_indices]
                    y=y[random_indices]
                    results=theilslopes(y,x)
                    slopes[i,j]=results[0]
                    slopesm[i,j]=results[2]
                    slopesp[i,j]=results[3]
                    
    return slopes,slopesm,slopesp


def calculate_slopes_per_fish_with_simple(params_list,deltavs_trials_list,strengthgradients_trials_list,min_quantile=0.1,max_quantile=0.9):
    params=params_list[0]
    slopes=np.full((len(params_list),len(params)),np.nan)
    slopesm=np.full((len(params_list),len(params)),np.nan)
    slopesp=np.full((len(params_list),len(params)),np.nan)
    for j in tqdm(range(len(params))):
        for i in range(len(params_list)):
            if len(deltavs_trials_list[i][j])>0:
                x=np.concatenate(deltavs_trials_list[i][j])
                y=np.concatenate(strengthgradients_trials_list[i][j])
                mask=(x>np.quantile(x,min_quantile))&(x<np.quantile(x,max_quantile))
                if np.any(mask):
                    x=x[mask]
                    y=y[mask]
                    slopes[i,j]=factor_that_minimizes_mse(x,y)
                    slopesm[i,j]=0
                    slopesp[i,j]=0
                    
    return slopes,slopesm,slopesp
                
def calculate_gs_from_slopes(slopes,params,mode,tau=None,alpha=None,target=None):
    if mode=='alpha':
        gs=slopes*params*tau
    elif mode=='vc':
        gs=slopes*alpha*tau
    elif mode=='fixed_strength':
        gs=slopes*params/target*tau
    elif mode=='tau':
        gs=slopes*alpha*(tau+params)
    else:
        print('undefined mode')
    return gs
                


def get_peak(ts,impulse_resp,tau=0):
    imax=np.argmax(np.cumsum(impulse_resp))
    kernel_times=ts[:imax+1]+tau
    kernel_points=impulse_resp[:imax+1]
    kernel_times=np.insert(kernel_times,0,[0,tau])
    kernel_points=np.insert(kernel_points,0,[0,0])
    return kernel_times,kernel_points


def extract_drift_and_diffusion(params_list,deltavs,deltavs_trials_list,strengthgradients_trials_list,fs,n_bins=50,uncertainty='mas',sqrtn=False,adaptive_binning=False):
    params=params_list[0]
    drifts=np.full((len(params_list),len(params),n_bins),np.nan)
    diffusions=np.full((len(params_list),len(params),n_bins),np.nan)
    vs_bins=np.zeros((len(params),n_bins))

    for j in range(len(params)):
        if adaptive_binning:
            percentiles=np.linspace(5,95,n_bins+1)
            bin_edges=np.percentile(deltavs[j],percentiles)        
            vs_bins[j]=(bin_edges[1:]+bin_edges[:-1])/2
            for i in tqdm(range(len(params_list))):
                if len(deltavs_trials_list[i][j])>0:
                    x=np.concatenate(deltavs_trials_list[i][j])
                    y=np.concatenate(strengthgradients_trials_list[i][j])
                    for k in range(len(vs_bins[j])):
                        mask=(x<bin_edges[k+1])&(x>bin_edges[k])
                        drifts[i,j,k]=np.nanmean(y[mask])
                        diffusions[i,j,k]=np.sqrt(np.nanmean(y[mask]**2)/fs)


        else:
            vmin=np.nanquantile(deltavs[j],0.05)
            vmax=np.nanquantile(deltavs[j],0.95)
            vs_bins[j]=np.linspace(vmin,vmax,n_bins)
            dv=vs_bins[j,1]-vs_bins[j,0]
            for i in tqdm(range(len(params_list))):
                if len(deltavs_trials_list[i][j])>0:
                    x=np.concatenate(deltavs_trials_list[i][j])
                    y=np.concatenate(strengthgradients_trials_list[i][j])
                    for k,vs in enumerate(vs_bins[j]):
                        mask=(x<vs+dv)&(x>vs-dv)
                        drifts[i,j,k]=np.nanmean(y[mask])
                        diffusions[i,j,k]=np.sqrt(np.nanmean(y[mask]**2)/fs)

    drifts_median=np.nanmedian(drifts,axis=0)
    diffusions_median=np.nanmedian(diffusions,axis=0)
    
    if uncertainty=='mas':
        drifts_se_median=median_abs_deviation(drifts,axis=0,nan_policy='omit')
        diffusions_se_median=median_abs_deviation(diffusions,axis=0,nan_policy='omit')
    elif uncertainty=='se_median':
        drifts_interquartile=np.nanquantile(drifts,0.75,axis=0)-np.nanquantile(drifts,0.25,axis=0)
        drifts_se_median=0.93*drifts_interquartile
        diffusions_interquartile=np.nanquantile(diffusions,0.75,axis=0)-np.nanquantile(diffusions,0.25,axis=0)
        diffusions_se_median=0.93*diffusions_interquartile
    elif uncertainty=='std':
        drifts_se_median=np.nanstd(drifts,axis=0)
        diffusions_se_median=np.nanstd(diffusions,axis=0)
    else:
        print('undefined uncertainty')
    
    if sqrtn:
        drifts_se_median/=np.sqrt(len(params_list))
        diffusions_se_median/=np.sqrt(len(params_list))

    return vs_bins,drifts_median,drifts_se_median,diffusions_median,diffusions_se_median


def calculate_weigthed_average_diffusions(deltavs,vs_bins,diffusions_median,diffusions_se_median):
    diffusions_avg=np.zeros(len(deltavs))
    diffusions_se_avg=np.zeros(len(deltavs))
    for i in range(len(deltavs)):
        dv=vs_bins[i,1]-vs_bins[i,0]
        weights=np.histogram(deltavs[i],bins=np.append(vs_bins[i]-dv,vs_bins[i,-1]+dv))[0]
        diffusions_avg[i]=np.nansum(weights*diffusions_median[i])/np.sum(weights)
        diffusions_se_avg[i]=np.nansum(weights*diffusions_se_median[i])/np.sum(weights)
        #diffusions_se_avg[i]=np.sqrt(np.nansum(weights**2*diffusions_se_median[i]**2)/np.sum(weights)**2)
    return diffusions_avg,diffusions_se_avg



def retransform_tail_points(tail_points,cam_ps,track_ps):
    '''
    go back from body-axis-aligned reference (cm) to image reference (pix)
    '''
    tail_points_retransformed=np.zeros_like(tail_points)
    for i in range(len(tail_points)):
        tail_points_retransformed[i]=(track_ps['R'].T)@tail_points[i]
        tail_points_retransformed[i,1]*=-1
        tail_points_retransformed[i]=tail_points_retransformed[i]/cam_ps['cm_per_pix']+track_ps['initial_pos']
    return tail_points_retransformed


def calculate_reaction_times(exp,prot_ps,threshold_strength=None):
    '''
    calculate time from beginning of trials to beginning of swimming
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    
    reaction_times=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(reaction_times)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
        indices_swimming=np.nonzero(np.abs(exp['strengths'][cameratimes_trial])>=threshold_strength)[0]
        if len(indices_swimming)==0:
            reaction_times[trial]=tmax-tmin
        else:
            reaction_times[trial]=exp['cameratimes'][cameratimes_trial][indices_swimming[0]]-tmin
    return reaction_times


def get_reaction_times_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=0.05):
    
    parameter_trials=prot_ps[parameter_trials_str]

    params=np.unique(parameter_trials)
    reaction_times=[[] for i in range(len(params))]
    for i,param in enumerate(params):
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            indices_swimming=np.nonzero(np.abs(exp['strengths'][cameratimes_trial])>=threshold_strength)[0]
            if len(indices_swimming)==0:
                reaction_times[i].append(tmax-tmin)
            else:
                reaction_times[i].append(exp['cameratimes'][cameratimes_trial][indices_swimming[0]]-tmin)

    return reaction_times,params


def get_reaction_times_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05):
    
    reaction_times_list=[]
    params_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        reaction_times,params=get_reaction_times_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=threshold_strength)
        
                
        reaction_times_list.append(reaction_times)
        params_list.append(params)
    

    return reaction_times_list,params_list







def get_drifts_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=0.05,segment_duration=5,segment_overlap=0.5,max_strength=10,min_sequence_length=5,sigma_strength=0.06,normalize=True):
    
    parameter_trials=prot_ps[parameter_trials_str]

    params=np.unique(parameter_trials)
    drifts=[[] for i in range(len(params))]
    
    window_size=round(segment_duration*cam_ps['acquisition_rate'])
    
    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma_strength)

    for i,param in enumerate(params):
        sequences=[]
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])

            strengths_masked=strengths_filt[mask]
            consecutive_sequences=find_consecutive_sequences_mask(strengths_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
            for seq in consecutive_sequences:
                if len(seq)>=round(min_sequence_length*cam_ps['acquisition_rate']):
                    sequences+=seq
                    
            if sequences and len(sequences)>round(segment_duration*cam_ps['acquisition_rate']):    
                if normalize:
                    drifts[i]+=list(-1/prot_ps['v_flow_trials'][trial]*(prot_ps['v_flow_trials'][trial]-prot_ps['alpha_trials'][trial]*np.array(rolling_mean(np.array(sequences),window_size,overlap_fraction=segment_overlap))))
                else:
                    drifts[i]+=list(-(prot_ps['v_flow_trials'][trial]-prot_ps['alpha_trials'][trial]*np.array(rolling_mean(np.array(sequences),window_size,overlap_fraction=segment_overlap))))
                    

    return drifts,params


def get_drifts_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05,segment_duration=5,segment_overlap=0.5,max_strength=10,min_sequence_length=5,sigma_strength=0.06):
    
    drifts_list=[]
    params_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        drifts,params=get_drifts_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=threshold_strength,segment_duration=segment_duration,
                                               segment_overlap=segment_overlap,max_strength=max_strength,min_sequence_length=min_sequence_length,sigma_strength=sigma_strength)
        
                
        drifts_list.append(drifts)
        params_list.append(params)
    

    return drifts_list,params_list



def get_swimming_fractions_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=0.05,discard_reaction_time=False):
    
    parameter_trials=prot_ps[parameter_trials_str]

    params=np.unique(parameter_trials)
    swimming_fractions=[[] for i in range(len(params))]
    

    for i,param in enumerate(params):
        sequences=[]
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            indices_swimming=np.nonzero(np.abs(exp['strengths'][mask])>=threshold_strength)[0]
            if len(indices_swimming)==0:
                swimming_fractions[i].append(0)
            else:
                if discard_reaction_time:
                    reaction_time=(exp['cameratimes'][mask][indices_swimming[0]]-tmin)
                    mask=tuple([(exp['cameratimes']>tmin+reaction_time)&(exp['cameratimes']<tmax)])

                swimming_fractions[i].append(np.sum(np.abs(exp['strengths'][mask])>=threshold_strength)/len(exp['strengths'][mask]))

            
    return swimming_fractions,params
    

def get_swimming_fractions_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05,discard_reaction_time=False):
    
    swimming_fractions_list=[]
    params_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        swimming_fractions,params=get_swimming_fractions_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=threshold_strength,discard_reaction_time=discard_reaction_time)
        
                
        swimming_fractions_list.append(swimming_fractions)
        params_list.append(params)
    

    return swimming_fractions_list,params_list





def get_swimming_durations_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=0.05,sigma=0.03):
    
    parameter_trials=prot_ps[parameter_trials_str]

    params=np.unique(parameter_trials)
    swimming_durations=[[] for i in range(len(params))]
    
    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma)

    for i,param in enumerate(params):
        sequences=[]
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            mask_swimming=np.abs(strengths_filt[mask])>=threshold_strength

            for key, group in groupby(mask_swimming):
                if key==1:
                    swimming_durations[i].append(len(list(group))/cam_ps['acquisition_rate'])
            
    return swimming_durations,params
    

    
def get_swimming_durations_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05,sigma=0.03):
    
    swimming_durations_list=[]
    params_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        swimming_durations,params=get_swimming_durations_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=threshold_strength,sigma=sigma)
        
                
        swimming_durations_list.append(swimming_durations)
        params_list.append(params)
    

    return swimming_durations_list,params_list


def calculate_position_errors(exp,prot_ps,reaction_times):
    '''
    calculate standard deviation of virtual fish position from when it starts swimming to the end of the trial
    '''
    position_errors=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(position_errors)):
        if reaction_times[trial]<prot_ps['flow_duration_trials'][trial]:
            tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            updatetimes_trial=tuple([(exp['updatetimes']>tmin)&(exp['updatetimes']<tmax)])
            position_errors[trial]=np.std(np.linalg.norm(exp['phases'][updatetimes_trial],axis=-1))*prot_ps['spatial_period_trials'][trial]/(prot_ps['v_flow_trials'][trial]*(tmax-tmin))*np.sqrt(12)
        else:
            position_errors[trial]=np.nan
    return position_errors
    

def calculate_speed_errors(exp,prot_ps,proj_ps,reaction_times,sigma_strength=0.025):
    '''
    calculate root mean square of net flow speed from beginning of swimming to trial end
    '''
    v_feedback_filt=gaussian_filter(exp['v_feedback'],sigma_strength*proj_ps['refresh_rate'])
    speed_errors=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(speed_errors)):
        if reaction_times[trial]<prot_ps['flow_duration_trials'][trial]:
            tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            updatetimes_trial=tuple([(exp['updatetimes']>tmin)&(exp['updatetimes']<tmax)])
            speed_errors[trial]=np.sqrt(np.mean((v_feedback_filt[updatetimes_trial]-exp['v_flow'][updatetimes_trial])**2))
        else:
            speed_errors[trial]=np.nan
    return speed_errors


def calculate_speed_differences(exp,prot_ps,cam_ps,threshold_strength=None,discard_non_swimming_time=True,relative=True):
    '''
    calculate average net flow speed
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    strength_filt=get_strengths_filt(exp,cam_ps)
    speed_differences=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(speed_differences)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
        if discard_non_swimming_time:
            cameratimes_trial=tuple(cameratimes_trial&(strength_filt>=threshold_strength))
        speed_differences[trial]=np.mean(prot_ps['alpha_trials'][trial]*strength_filt[cameratimes_trial]-prot_ps['v_flow_trials'][trial])
        if relative:
            speed_differences[trial]/=prot_ps['v_flow_trials'][trial]
    return speed_differences


def calculate_speed_rmses(exp,prot_ps,cam_ps,threshold_strength=None,discard_non_swimming_time=True,relative=True):
    '''
    calculate RMSE of fish speed with respect to current speed
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    strength_filt=get_strengths_filt(exp,cam_ps)
    speed_rmses=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(speed_rmses)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
        if discard_non_swimming_time:
            cameratimes_trial=tuple(cameratimes_trial&(strength_filt>=threshold_strength))
        speed_rmses[trial]=np.mean((prot_ps['alpha_trials'][trial]*strength_filt[cameratimes_trial]-prot_ps['v_flow_trials'][trial])**2)**0.5
        if relative:
            speed_rmses[trial]/=prot_ps['v_flow_trials'][trial]
    return speed_rmses


def calculate_accuracies(relative_differences):
    '''
    calculate 1-|relative_differences| and threshold to positive values
    '''
    accuracies=1-np.abs(relative_differences)
    accuracies[accuracies<0]=-1e-3
    return accuracies

def calculate_speed_accuracies(exp,prot_ps,proj_ps,reaction_times,sigma_v_feedback=0.025):
    '''
    calculate 1-|v_f-v_c|/v_c after fish started swimming
    '''
    v_feedback_filt=gaussian_filter(exp['v_feedback'],sigma_v_feedback*proj_ps['refresh_rate'])
    speed_accuracies=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(speed_accuracies)):
        if reaction_times[trial]<prot_ps['flow_duration_trials'][trial]:
            tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            updatetimes_trial=tuple([(exp['updatetimes']>tmin)&(exp['updatetimes']<tmax)])
            speed_accuracies[trial]=np.mean(1-np.abs(v_feedback_filt[updatetimes_trial]/exp['v_flow'][updatetimes_trial]-1))
    return speed_accuracies

def calculate_fractions_at_current_speed(exp,prot_ps,proj_ps,reaction_times,tolerance=0.1,sigma_v_feedback=0.025):
    '''
    calculate fraction of time close to current speed after fish started swimming
    '''
    v_feedback_filt=gaussian_filter(exp['v_feedback'],sigma_v_feedback*proj_ps['refresh_rate'])
    fractions_at_current_speed=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(fractions_at_current_speed)):
        if reaction_times[trial]<prot_ps['flow_duration_trials'][trial]:
            tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            updatetimes_trial=tuple([(exp['updatetimes']>tmin)&(exp['updatetimes']<tmax)])
            fractions_at_current_speed[trial]=np.sum(np.abs(v_feedback_filt[updatetimes_trial]/exp['v_flow'][updatetimes_trial]-1)<=tolerance)/len(v_feedback_filt[updatetimes_trial]) 
    return fractions_at_current_speed
    

def calculate_swimming_fractions(exp,prot_ps,threshold_strength=None,discard_reaction_time=False):
    '''
    calculate fraction of time spent swimming from beginning of swimming to trial end
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    reaction_times=calculate_reaction_times(exp,prot_ps,threshold_strength=threshold_strength)
    swimming_fractions=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(swimming_fractions)):
        if reaction_times[trial]<prot_ps['flow_duration_trials'][trial]:
            if discard_reaction_time:
                tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
            else:
                tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            swimming_fractions[trial]=np.sum(np.abs(exp['strengths'][cameratimes_trial])>=threshold_strength)/len(exp['strengths'][cameratimes_trial])   
        else:
            swimming_fractions[trial]=0
    return swimming_fractions


def calculate_orientation_errors(exp,prot_ps,fraction_of_trial_considered=0.2):
    '''
    calculate root mean square of the fish orientation relative to that of the flow on an ending fraction of the trial
    '''
    orientation_errors=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(orientation_errors)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        tmin=tmax-fraction_of_trial_considered*prot_ps['flow_duration_trials'][trial]
        updatetimes_trial=tuple([(exp['updatetimes']>tmin)&(exp['updatetimes']<tmax)])
        orientation_errors[trial]=np.sqrt(np.mean((exp['oris'][updatetimes_trial]-(exp['v_flow_angle'][updatetimes_trial]+exp['oris_pause'][updatetimes_trial]))**2))
    return orientation_errors

def calculate_orientation_differences(exp,prot_ps,fraction_of_trial_considered=0.2):
    '''
    calculate difference of the fish orientation relative to that of the flow on an ending fraction of the trial
    '''
    orientation_differences=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(orientation_differences)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        tmin=tmax-fraction_of_trial_considered*prot_ps['flow_duration_trials'][trial]
        updatetimes_trial=tuple([(exp['updatetimes']>tmin)&(exp['updatetimes']<tmax)])
        orientation_differences[trial]=np.mean(exp['oris'][updatetimes_trial]-(exp['v_flow_angle'][updatetimes_trial]+exp['oris_pause'][updatetimes_trial]))
    return orientation_differences
    

def calculate_starting_average_strengths(exp,prot_ps,threshold_strength=None,duration=0.5):
    '''
    calculate average swimming strength at the beginning of swimming
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    reaction_times=calculate_reaction_times(exp,prot_ps,threshold_strength=threshold_strength)
    starting_average_strengths=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(starting_average_strengths)):
        if reaction_times[trial]<prot_ps['flow_duration_trials'][trial]:
            tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
            tmax=tmin+duration
            cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            starting_average_strengths[trial]=np.mean(exp['strengths'][cameratimes_trial])
        else:
            starting_average_strengths[trial]=np.nan
    return starting_average_strengths


def get_starting_average_strengths_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=0.05,duration=0.5,get_turning_strength=False,get_distances=False):
    
    parameter_trials=prot_ps[parameter_trials_str]

    params=np.unique(parameter_trials)
    starting_average_strengths=[[] for i in range(len(params))]
    if get_distances:
        distances=[[] for i in range(len(params))]
    
    reaction_times=calculate_reaction_times(exp,prot_ps,threshold_strength=threshold_strength)

    for i,param in enumerate(params):
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            if reaction_times[trial]<prot_ps['flow_duration_trials'][trial] and reaction_times[trial]>0.05:

                tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
                tmax=tmin+duration
                cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
                if get_turning_strength:
                    starting_average_strengths[i].append(np.mean(exp['turning_strengths'][cameratimes_trial]))
                else:
                    starting_average_strengths[i].append(np.mean(exp['strengths'][cameratimes_trial]))

            else:
                starting_average_strengths[i].append(np.nan)
                
            if get_distances:
                distances[i].append(reaction_times[trial]*prot_ps['v_flow_trials'][trial])
    
    if get_distances:
        return starting_average_strengths,distances,params
    else:
        return starting_average_strengths,params


def get_starting_average_strengths_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05,duration=0.5,get_turning_strength=False,get_distances=False):
    
    starting_average_strengths_list=[]
    params_list=[]
    if get_distances:
        distances_list=[]

    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        if get_distances:
            starting_average_strengths,distances,params=get_starting_average_strengths_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=threshold_strength,duration=duration,get_turning_strength=get_turning_strength,get_distances=get_distances)
            distances_list.append(distances)
        else:
                        starting_average_strengths,params=get_starting_average_strengths_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=threshold_strength,duration=duration,get_turning_strength=get_turning_strength,get_distances=get_distances)


        starting_average_strengths_list.append(starting_average_strengths)
        params_list.append(params)
    
    if get_distances:
        return starting_average_strengths_list,distances_list,params_list
    else:
        return starting_average_strengths_list,params_list


def get_starting_average_strengths_during_trials_two_params(exp,cam_ps,prot_ps,parameter_trials_str1,parameter_trials_str2,threshold_strength=0.05,duration=0.5,get_turning_strength=False):
    
    parameter_trials1=prot_ps[parameter_trials_str1]
    parameter_trials2=prot_ps[parameter_trials_str2]

    params1=np.unique(parameter_trials1)
    params2=np.unique(parameter_trials2)
    starting_average_strengths=[[[] for j in range(len(params2))] for i in range(len(params1))]
    
    reaction_times=calculate_reaction_times(exp,prot_ps,threshold_strength=threshold_strength)

    for i,param1 in enumerate(params1):
        for j,param2 in enumerate(params2):
            selected_trials=np.nonzero((parameter_trials1==param1)&(parameter_trials2==param2))[0]

            for trial in selected_trials:
                if reaction_times[trial]<prot_ps['flow_duration_trials'][trial] and reaction_times[trial]>0.05:

                    tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
                    tmax=tmin+duration
                    cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
                    if get_turning_strength:
                        starting_average_strengths[i][j].append(np.mean(exp['turning_strengths'][cameratimes_trial]))
                    else:
                        starting_average_strengths[i][j].append(np.mean(exp['strengths'][cameratimes_trial]))

                else:
                    starting_average_strengths[i][j].append(np.nan)

    return starting_average_strengths,params1,params2


def get_starting_average_strengths_multiple_experiments_two_params(experiments,parameter_trials_str1,parameter_trials_str2,threshold_strength=0.05,duration=0.5,get_turning_strength=False):
    
    starting_average_strengths_list=[]
    params1_list=[]
    params2_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        starting_average_strengths,params1,params2=get_starting_average_strengths_during_trials_two_params(exp,cam_ps,prot_ps,parameter_trials_str1,parameter_trials_str2,threshold_strength=threshold_strength,duration=duration,get_turning_strength=get_turning_strength)
        
                
        starting_average_strengths_list.append(starting_average_strengths)
        params1_list.append(params1)
        params2_list.append(params2)
    

    return starting_average_strengths_list,params1_list,params2_list



def calculate_average_strengths(exp,prot_ps,threshold_strength=None,discard_non_swimming_time=True):
    '''
    calculate average swimming strength over the trials
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    reaction_times=calculate_reaction_times(exp,prot_ps,threshold_strength=threshold_strength)
    average_strengths=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(average_strengths)):
        if reaction_times[trial]<prot_ps['flow_duration_trials'][trial]:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            if discard_non_swimming_time:
                average_strengths[trial]=np.mean(exp['strengths'][cameratimes_trial][np.abs(exp['strengths'][cameratimes_trial])>=threshold_strength])
            else:
                average_strengths[trial]=np.mean(exp['strengths'][cameratimes_trial])
        else:
            average_strengths[trial]=np.nan
    return average_strengths

def calculate_average_turning_strengths(exp,prot_ps,threshold_strength=None,discard_non_swimming_time=True):
    '''
    calculate average turning strength over the trials
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    reaction_times=calculate_reaction_times(exp,prot_ps,threshold_strength=threshold_strength)
    average_turning_strengths=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(average_turning_strengths)):
        if reaction_times[trial]<prot_ps['flow_duration_trials'][trial]:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            if discard_non_swimming_time:
                average_turning_strengths[trial]=np.mean(exp['turning_strengths'][cameratimes_trial][np.abs(exp['strengths'][cameratimes_trial])>=threshold_strength])
            else:
                average_turning_strengths[trial]=np.mean(exp['turning_strengths'][cameratimes_trial])
        else:
            average_turning_strengths[trial]=np.nan
    return average_turning_strengths


def calculate_statistics_over_trials(exp,prot_ps,metric,parameter_trials):
    '''
    calculate mean and standard deviation of a metric over the trials
    '''
    parameter_trials=np.array(parameter_trials)
    if len(parameter_trials.shape)==1:
        parameter_values=np.sort(np.unique(parameter_trials))
        metric_mean=np.zeros(len(parameter_values))
        metric_std=np.zeros(len(parameter_values))
        for i in range(len(parameter_values)):
            trials=tuple([parameter_trials==parameter_values[i]])
            metric_mean[i]=np.mean(metric[trials])
            metric_std[i]=np.std(metric[trials])
        return parameter_values,metric_mean,metric_std
    elif len(parameter_trials.shape)==2:
        parameter_values=[np.sort(np.unique(parameter_trials[i])) for i in range(len(parameter_trials))]
        metric_mean=np.zeros([len(parameter_values[i]) for i in range(len(parameter_values))])
        metric_std=np.zeros([len(parameter_values[i]) for i in range(len(parameter_values))])
        for parameter_indices,_ in np.ndenumerate(metric_mean):
            trials_for_each_parameter=[parameter_trials[i]==parameter_values[i][parameter_indices[i]] for i in range(len(parameter_indices))]
            trials_all_parameters=np.all(trials_for_each_parameter,axis=0)
            metric_mean[parameter_indices]=np.mean(metric[trials_all_parameters])
            metric_std[parameter_indices]=np.std(metric[trials_all_parameters])
        return parameter_values,metric_mean,metric_std
    else:
        print('parameter_trials has to be 1d or have only 2 rows')



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



def calculate_oscillation_periods_mode(exp,prot_ps,cam_ps,threshold_strength=None,sigma_signal=0.05,sigma_spectrum_hz=0.05):
    '''
    calculate the period of the oscillations by taking the mode of the fourier spectrum
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    
    sigma_signal=sigma_signal*cam_ps['acquisition_rate']
    filtered_signal=gaussian_filter(exp['turning_strengths'],sigma_signal)
    
    oscillation_periods=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(oscillation_periods)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_swimming=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)&(np.abs(exp['strengths'])>=threshold_strength)])
        if np.sum(cameratimes_swimming)>0:
            signal_selected=filtered_signal[cameratimes_swimming]
            signal_selected-=np.mean(signal_selected)
            power=np.abs(fft.rfft(signal_selected))**2
            frequencies=fft.rfftfreq(len(signal_selected),1/cam_ps['acquisition_rate'])
            sigma_spectrum=sigma_spectrum_hz/frequencies[1]
            filtered_power=gaussian_filter(power,sigma_spectrum)    
            oscillation_periods[trial]=1/frequencies[np.argmax(filtered_power)]
        else:
            oscillation_periods[trial]=np.nan
            
    return oscillation_periods


def calculate_oscillation_periods_welch(exp,prot_ps,cam_ps,threshold_strength=None,nperseg=None,sigma_signal=0.05,sigma_spectrum_hz=0.05):
    '''
    calculate the period of the oscillations by taking the mode of the fourier spectrum, using the welch algorithm
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    if nperseg is None:
         nperseg=10*cam_ps['acquisition_rate']
            
    sigma_signal=sigma_signal*cam_ps['acquisition_rate']
    filtered_signal=gaussian_filter(exp['turning_strengths'],sigma_signal)
    
    oscillation_periods=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(oscillation_periods)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_swimming=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)&(np.abs(exp['strengths'])>=threshold_strength)])
        if np.sum(cameratimes_swimming)>0:
            signal_selected=filtered_signal[cameratimes_swimming]
            signal_selected-=np.mean(signal_selected)      
            frequencies,power=welch(signal_selected,cam_ps['acquisition_rate'],nperseg=nperseg)

            sigma_spectrum=sigma_spectrum_hz/frequencies[1]
            filtered_power=gaussian_filter(power,sigma_spectrum)    
            oscillation_periods[trial]=1/frequencies[np.argmax(filtered_power)]
        else:
            oscillation_periods[trial]=np.nan
            
    return oscillation_periods



def calculate_stds_strength_change(exp,prot_ps,cam_ps,threshold_strength=None,sigma_strength=0.025):
    '''
    wrapper to calculate standard deviation of the rate of change of the swimming strength
    '''
    strengths_change=np.gradient(exp['strengths'],1/cam_ps['acquisition_rate'])
    return calculate_stds(strengths_change,exp,prot_ps,cam_ps,threshold_strength=threshold_strength,sigma_signal=sigma_strength)


def calculate_stds_turning_strength(exp,prot_ps,cam_ps,threshold_strength=None,sigma_turning_strength=0.05):
    '''
    wrapper to calculate standard deviation of the rate of change of the turning strength
    '''
    return calculate_stds(exp['turning_strengths'],exp,prot_ps,cam_ps,threshold_strength=threshold_strength,sigma_signal=sigma_turning_strength)
    

def calculate_stds(signal,exp,prot_ps,cam_ps,threshold_strength=None,sigma_signal=0.05,sigma_strength=0.025):
    '''
    calculate standard deviation of a signal considering the times when the fish is swimming
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    
    sigma_signal=sigma_signal*cam_ps['acquisition_rate']
    filtered_signal=gaussian_filter(signal,sigma_signal)
    strengths_filt=gaussian_filter(exp['strengths'],sigma_strength*cam_ps['acquisition_rate'])
    
    stds=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(stds)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_swimming=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)&(np.abs(strengths_filt>=threshold_strength))])
        if np.sum(cameratimes_swimming)>0:
            signal_selected=filtered_signal[cameratimes_swimming]
            stds[trial]=np.std(signal_selected)
        else:
            stds[trial]=np.nan
            
    return stds


def calculate_hilbert_strength_change(exp,prot_ps,cam_ps,threshold_strength=None,sigma_strength=0.025):
    '''
    wrapper to calculate hilbert transform of the rate of change of the swimming strength
    '''
    strengths_change=np.gradient(exp['strengths'],1/cam_ps['acquisition_rate'])
    return calculate_hilbert(strengths_change,exp,prot_ps,cam_ps,threshold_strength=threshold_strength,sigma_signal=sigma_strength)


def calculate_hilbert_turning_strength(exp,prot_ps,cam_ps,threshold_strength=None,sigma_turning_strength=0.05):
    '''
    wrapper to calculate hilbert transform of the rate of change of the turning strength
    '''
    return calculate_hilbert(exp['turning_strengths'],exp,prot_ps,cam_ps,threshold_strength=threshold_strength,sigma_signal=sigma_turning_strength)
    

def calculate_hilbert(signal,exp,prot_ps,cam_ps,threshold_strength=None,sigma_signal=0.05,sigma_strength=0.025):
    '''
    calculate hilbert transform of a signal
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    
    sigma_signal=sigma_signal*cam_ps['acquisition_rate']
    filtered_signal=gaussian_filter(signal,sigma_signal)
    strengths_filt=gaussian_filter(exp['strengths'],sigma_strength*cam_ps['acquisition_rate'])
    
    amplitudes=np.zeros(len(prot_ps['flow_start_times']))
    frequencies=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(amplitudes)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_swimming=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)&(np.abs(strengths_filt>=threshold_strength))])
        if np.sum(cameratimes_swimming)>0:
            signal_selected=filtered_signal[cameratimes_swimming]

            analytical_signal = hilbert(signal_selected)
            ampl=np.abs(analytical_signal)
            freq=1/(2*np.pi)*np.gradient(np.unwrap(np.angle(analytical_signal)),1/cam_ps['acquisition_rate'])
            
            amplitudes[trial]=np.mean(ampl)
            frequencies[trial]=np.mean(freq)
        else:
            amplitudes[trial]=np.nan
            frequencies[trial]=np.nan
            
    return amplitudes,frequencies


def calculate_periods_quantiles_strength_change(exp,prot_ps,cam_ps,threshold_strength=None,sigma_strength=0.025):
    '''
    wrapper to calculate oscillation period of the rate of change of the swimming strength
    '''
    strengths_change=np.gradient(exp['strengths'],1/cam_ps['acquisition_rate'])
    return calculate_periods_quantiles(strengths_change,exp,prot_ps,cam_ps,threshold_strength=threshold_strength,sigma_signal=sigma_strength)


def calculate_periods_quantiles_turning_strength(exp,prot_ps,cam_ps,threshold_strength=None,sigma_turning_strength=0.05):
    '''
    wrapper to calculate oscillation period of the rate of change of the turning strength
    '''
    return calculate_periods_quantiles(exp['turning_strengths'],exp,prot_ps,cam_ps,threshold_strength=threshold_strength,sigma_signal=sigma_turning_strength)
    

def calculate_periods_quantiles(signal,exp,prot_ps,cam_ps,threshold_strength=None,sigma_signal=0.05,sigma_strength=0.025):
    '''
    wrapper to calculate oscillation period of a signal by evaluating the quantiles of the fourier spectrum
    '''
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    
    sigma_signal=sigma_signal*cam_ps['acquisition_rate']
    filtered_signal=gaussian_filter(signal,sigma_signal)
    strengths_filt=gaussian_filter(exp['strengths'],sigma_strength*cam_ps['acquisition_rate'])
    
    periods_median=np.zeros(len(prot_ps['flow_start_times']))
    periods_lower=np.zeros(len(prot_ps['flow_start_times']))
    periods_upper=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(periods_median)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_swimming=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)&(np.abs(strengths_filt>=threshold_strength))])
        if np.sum(cameratimes_swimming)>0:
            signal_selected=filtered_signal[cameratimes_swimming]
            signal_selected-=np.mean(signal_selected)
            power=np.abs(fft.rfft(signal_selected))**2
            frequencies=fft.rfftfreq(len(signal_selected),1/cam_ps['acquisition_rate'])
            
            cum=np.cumsum(power)/np.sum(power)
            fmedian=frequencies[np.argmin(np.abs(cum-0.5))]
            fupper=frequencies[np.argmin(np.abs(cum-0.75))]
            flower=frequencies[np.argmin(np.abs(cum-0.25))]

            periods_median[trial]=1/fmedian
            periods_lower[trial]=1/fupper
            periods_upper[trial]=1/flower
        else:
            periods_median[trial]=np.nan
            periods_lower[trial]=np.nan
            periods_upper[trial]=np.nan
            
    return periods_median,periods_lower,periods_upper


def get_v_flow_angle_nan(exp):
    v_flow_angle_nan=np.copy(exp['v_flow_angle'])
    v_flow_angle_nan[exp['v_flow']==0]=np.nan
    return v_flow_angle_nan

def get_v_feedback_filt(exp,proj_ps,sigma=None):
    if sigma is None:
        sigma=2/proj_ps['refresh_rate']
    v_feedback_filt=gaussian_filter(exp['v_feedback'],sigma*proj_ps['refresh_rate'])
    return v_feedback_filt

def get_strengths_filt(exp,cam_ps,sigma=0.03):
    strengths_filt=gaussian_filter(exp['strengths'],sigma*cam_ps['acquisition_rate'])
    return strengths_filt

def get_turning_strengths_filt(exp,cam_ps,sigma=0.06):
    turning_strengths_filt=gaussian_filter(exp['turning_strengths'],sigma*cam_ps['acquisition_rate'])
    return turning_strengths_filt

def get_omega_feedback_filt(exp,proj_ps,sigma=None):
    if sigma is None:
        sigma=2/proj_ps['refresh_rate']
    omega_feedback_filt=gaussian_filter(exp['omega_feedback'],sigma*proj_ps['refresh_rate'])
    return omega_feedback_filt



#def get_strengths_during_trials(exp,prot_ps,parameter_trials,threshold_strength,reaction_times=None):
#    '''
#    takes in one run and a list of parameters corresponding to each trial, then one gets a list with the segments of strength between
#    the reaction time and the end of the trial for each unique value of the parameter
#    '''
#    if reaction_times is None:
#        reaction_times=np.zeros(len(parameter_trials))
#    params=np.unique(parameter_trials)
#    strengths=[]
#    for param in params:
#        strengths_param=[]
#        for trial in range(len(parameter_trials)):
#            if parameter_trials[trial]==param:
#                tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
#                tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
#                cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
#                strengths_param+=list(exp['strengths'][cameratimes_trial])
#        strengths.append(strengths_param)
#    return strengths,params

def get_strengths_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05,return_alpha=False,return_target_strength=False,return_v=False,discard_reaction_times=False,sigma=None):
    
    strengths_list=[]
    params_list=[]
    alpha_list=[]
    target_strength_list=[]
    v_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        if return_alpha:
            alpha_list.append(prot_ps['alpha_trials'][0])
        if return_target_strength:
            target_strength_list.append(prot_ps['v_flow_trials'][0]/prot_ps['alpha_trials'][0])
        if return_v:
            v_list.append(prot_ps['v_flow_trials'][0])
        parameter_trials=prot_ps[parameter_trials_str]
        strengths,params=get_strengths_during_trials(exp,cam_ps,prot_ps,parameter_trials,threshold_strength=threshold_strength,
                                                     discard_reaction_times=discard_reaction_times,sigma=sigma)    
            
        strengths_list.append(strengths)
        params_list.append(params)
    
    if return_alpha:
        return strengths_list,params_list,alpha_list
    elif return_target_strength:
        return strengths_list,params_list,target_strength_list
    elif return_v:
        return strengths_list,params_list,v_list
    else:
        return strengths_list,params_list


def get_strengths_during_trials(exp,cam_ps,prot_ps,parameter_trials,threshold_strength=0.05,discard_reaction_times=False,sigma=None):
    '''
    takes in one run and a list of parameters corresponding to each trial, then one gets a list with the segments of strength (after gaussian convolution) between
    the reaction time and the end of the trial for each unique value of the parameter
    '''
    if discard_reaction_times:
        reaction_times=calculate_reaction_times(exp,prot_ps,threshold_strength=threshold_strength)
    else:
        reaction_times=np.zeros(len(parameter_trials))

    if sigma is None:
        strengths_filt=exp['strengths']
    else:
        strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma)
    params=np.unique(parameter_trials)
    strengths=[]
    for param in params:
        strengths_param=[]
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]+reaction_times[trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            strengths_param+=list(strengths_filt[cameratimes_trial])
        strengths.append(strengths_param)
    return strengths,params


def get_turning_strengths_during_trials(exp,cam_ps,prot_ps,parameter_trials,sigma=None,get_orientation_diffs=False):

    if sigma is None:
        turning_strengths_filt=exp['turning_strengths']
    else:
        turning_strengths_filt=get_turning_strengths_filt(exp,cam_ps,sigma=sigma)
        
    params=np.unique(parameter_trials)
    turning_strengths=[]
    
    
    if get_orientation_diffs:
        orientation_diffs=[]
        f_orientation_diff=interp1d(exp['updatetimes'],-exp['oris']+exp['v_flow_angle']+exp['oris_pause'],fill_value='extrapolate')
    
    for param in params:
        turning_strengths_param=[]
        selected_trials=np.nonzero(parameter_trials==param)[0]
        if get_orientation_diffs:
            orientation_diffs_param=[]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            turning_strengths_param+=list(turning_strengths_filt[cameratimes_trial])
            if get_orientation_diffs:
                orientation_diffs_param+=list(f_orientation_diff(exp['cameratimes'][cameratimes_trial]))

        turning_strengths.append(turning_strengths_param)
        if get_orientation_diffs:
            orientation_diffs.append(orientation_diffs_param)

    if get_orientation_diffs:
        return turning_strengths,orientation_diffs,params
    else:
        return turning_strengths,params



def get_turning_strengths_multiple_experiments(experiments,parameter_trials_str,sigma=None,get_orientation_diffs=False):
    
    turning_strengths_list=[]
    params_list=[]
    if get_orientation_diffs:
        orientation_diffs_list=[]

    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        parameter_trials=prot_ps[parameter_trials_str]
        if get_orientation_diffs:
            turning_strengths,orientation_diffs,params=get_turning_strengths_during_trials(exp,cam_ps,prot_ps,parameter_trials,sigma=sigma,get_orientation_diffs=get_orientation_diffs)    
            orientation_diffs_list.append(orientation_diffs)
        else:
            turning_strengths,params=get_turning_strengths_during_trials(exp,cam_ps,prot_ps,parameter_trials,sigma=sigma,get_orientation_diffs=get_orientation_diffs)    

        turning_strengths_list.append(turning_strengths)
        params_list.append(params)
    
    if get_orientation_diffs:
        return turning_strengths_list,orientation_diffs_list,params_list
    else:
        return turning_strengths_list,params_list


def get_frequencies_during_trials(exp,cam_ps,prot_ps,parameter_trials,threshold_strength=0.05,sigma=0.06,window_length=10,min_sequence_length=2,max_strength=10):
    '''
    takes in one run and a list of parameters corresponding to each trial, then one gets a list with the segments of strength (after gaussian convolution) between
    the reaction time and the end of the trial for each unique value of the parameter
    '''


    if sigma is None:
        strengths_filt=exp['strengths']
    else:
        strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma)
        
    strengths_change_filt=np.gradient(strengths_filt,1/cam_ps['acquisition_rate'])
    
        
        
    params=np.unique(parameter_trials)
    frequencies_trials=[]
    power_density_trials=[]
    amplitude_fit_trials=[]
    mean_fit_trials=[]
    std_fit_trials=[]
        
    for param in params:

        selected_trials=np.nonzero(parameter_trials==param)[0]
        sequences=[]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            strengths_masked=exp['strengths'][mask]
            strengths_change_filt_masked=strengths_change_filt[mask]
            sequences+=find_consecutive_sequences_mask(strengths_change_filt_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
        
        sequences=[sequence for sequence in sequences if len(sequence)>=round(min_sequence_length*cam_ps['acquisition_rate'])]

        if len(sequences)>0:
            sequence=np.concatenate(sequences)
        else:
            sequence=np.array(sequences)
            
        if len(sequence)>round(window_length*cam_ps['acquisition_rate']):

            frequencies,power_density,[amplitude_fit,mean_fit,std_fit]=calculate_and_fit_fourier_spectrum(sequence,cam_ps,window_length=window_length,
                                                                                                initial_guess=[10,1,1],return_spectrum=True)
               
        else: 
            frequencies,power_density,amplitude_fit,mean_fit,std_fit=np.nan,np.nan,np.nan,np.nan,np.nan,
            
        frequencies_trials.append(frequencies)
        power_density_trials.append(power_density)
        amplitude_fit_trials.append(amplitude_fit)
        mean_fit_trials.append(mean_fit)
        std_fit_trials.append(std_fit)
            
            
        
    return params,frequencies_trials,power_density_trials,amplitude_fit_trials,mean_fit_trials,std_fit_trials



def get_delayed_gradient_during_trials(exp,cam_ps,prot_ps,parameter_trials,mode,threshold_strength=0.05,sigma_flow=0.06,sigma_gradient=0.03,tau=0.15,min_sequence_length=2,max_strength=5):

    deltavs_trials=[]
    strengthgradients_trials=[]
    
    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma_flow)
    strengths_change_filt=np.gradient(get_strengths_filt(exp,cam_ps,sigma=sigma_gradient),1/cam_ps['acquisition_rate'])
    
    di=round(tau*cam_ps['acquisition_rate'])
        
    params=np.unique(parameter_trials)
    for i,param in enumerate(params):
        
        if mode=='alpha':
            alphai=params[i]
            vci=prot_ps['v_flow_trials'][0]
            ddi=0
        elif mode=='vc':
            alphai=prot_ps['alpha_trials'][0]
            vci=params[i]
            ddi=0
        elif mode=='fixed_strength':
            alphai=params[i]/(prot_ps['v_flow_trials'][0]/prot_ps['alpha_trials'][0])
            vci=params[i]
            ddi=0
        elif mode=='tau':
            alphai=prot_ps['alpha_trials'][0]
            vci=prot_ps['v_flow_trials'][0]
            ddi=round(params[i]*cam_ps['acquisition_rate'])

        selected_trials=np.nonzero(parameter_trials==param)[0]
        sequences_strengths=[]
        sequences_strengths_change=[]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            strengths_masked=exp['strengths'][mask]
            strengths_filt_masked=strengths_filt[mask]
            strengths_change_filt_masked=strengths_change_filt[mask]
            
            sequences_strengths+=find_consecutive_sequences_mask(strengths_filt_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
            sequences_strengths_change+=find_consecutive_sequences_mask(strengths_change_filt_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
        
        sequences_strengths=[sequence for sequence in sequences_strengths if len(sequence)>=round(min_sequence_length*cam_ps['acquisition_rate'])]
        sequences_strengths_change=[sequence for sequence in sequences_strengths_change if len(sequence)>=round(min_sequence_length*cam_ps['acquisition_rate'])]

        deltavs=[]
        strengthgradients=[]
        
        for i in range(len(sequences_strengths)):
          
            deltavs.append(vci-alphai*np.array(sequences_strengths[i])[:-di-ddi])
            strengthgradients.append(np.array(sequences_strengths_change[i])[di+ddi:])    
            
        deltavs_trials.append(deltavs)
        strengthgradients_trials.append(strengthgradients)
            
        
    return deltavs_trials,strengthgradients_trials,params


def get_delayed_gradient_multiple_experiments(experiments,parameter_trials_str,mode,threshold_strength=0.05,sigma_flow=0.06,sigma_gradient=0.03,tau=0.15,min_sequence_length=2,max_strength=5):
    
    deltavs_trials_list=[]
    strengthgradients_trials_list=[]
    params_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')
            
        parameter_trials=prot_ps[parameter_trials_str]
        deltavs_trials,strengthgradients_trials,params=get_delayed_gradient_during_trials(exp,cam_ps,prot_ps,parameter_trials,mode,threshold_strength=threshold_strength,
                                                                            sigma_flow=sigma_flow,sigma_gradient=sigma_gradient,tau=tau,
                                                                            min_sequence_length=min_sequence_length,max_strength=max_strength)
        
        deltavs_trials_list.append(deltavs_trials)
        strengthgradients_trials_list.append(strengthgradients_trials)
        params_list.append(params)
        
    return deltavs_trials_list,strengthgradients_trials_list,params_list




def get_delayed_gradient_during_trials_simple_diff(exp,cam_ps,prot_ps,parameter_trials,mode,log=False,threshold_strength=0.05,sigma_flow=0.06,tau=0.15,min_sequence_length=2,max_strength=5):

    deltavs_trials=[]
    strengthgradients_trials=[]
    
    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma_flow)
    if log:
        strengths_change=np.diff(np.append(np.log(exp['strengths']),0))*cam_ps['acquisition_rate']
    else:
        strengths_change=np.diff(np.append(exp['strengths'],0))*cam_ps['acquisition_rate']
    #strengths_change=np.diff(np.append(exp['strengths'],0))*cam_ps['acquisition_rate']

    di=round(tau*cam_ps['acquisition_rate'])
        
    params=np.unique(parameter_trials)
    for i,param in enumerate(params):
        
        if mode=='alpha':
            alphai=params[i]
            vci=prot_ps['v_flow_trials'][0]
            ddi=0
        elif mode=='vc':
            alphai=prot_ps['alpha_trials'][0]
            vci=params[i]
            ddi=0
        elif mode=='fixed_strength':
            alphai=params[i]/(prot_ps['v_flow_trials'][0]/prot_ps['alpha_trials'][0])
            vci=params[i]
            ddi=0
        elif mode=='tau':
            alphai=prot_ps['alpha_trials'][0]
            vci=prot_ps['v_flow_trials'][0]
            ddi=round(params[i]*cam_ps['acquisition_rate'])

        selected_trials=np.nonzero(parameter_trials==param)[0]
        sequences_strengths=[]
        sequences_strengths_change=[]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
            strengths_masked=exp['strengths'][mask]
            strengths_filt_masked=strengths_filt[mask]
            strengths_change_masked=strengths_change[mask]
            
            sequences_strengths+=find_consecutive_sequences_mask(strengths_filt_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
            
            sequences_strengths_change+=find_consecutive_sequences_mask(strengths_change_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
            #if log:
            #    sequences_strengths_change+=find_consecutive_sequences_mask(strengths_change_masked/strengths_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
            #else:
            #    sequences_strengths_change+=find_consecutive_sequences_mask(strengths_change_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))

        
        sequences_strengths=[sequence for sequence in sequences_strengths if len(sequence)>=round(min_sequence_length*cam_ps['acquisition_rate'])]
        sequences_strengths_change=[sequence for sequence in sequences_strengths_change if len(sequence)>=round(min_sequence_length*cam_ps['acquisition_rate'])]

        deltavs=[]
        strengthgradients=[]
        
        for i in range(len(sequences_strengths)):
          
            deltavs.append(vci-alphai*np.array(sequences_strengths[i])[:-di-ddi])
            strengthgradients.append(np.array(sequences_strengths_change[i])[di+ddi:])    
            
        deltavs_trials.append(deltavs)
        strengthgradients_trials.append(strengthgradients)
            
        
    return deltavs_trials,strengthgradients_trials,params


def get_delayed_gradient_multiple_experiments_simple_diff(experiments,parameter_trials_str,mode,log=False,threshold_strength=0.05,sigma_flow=0.06,tau=0.15,min_sequence_length=2,max_strength=5):
    
    deltavs_trials_list=[]
    strengthgradients_trials_list=[]
    params_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')
            
        parameter_trials=prot_ps[parameter_trials_str]
        deltavs_trials,strengthgradients_trials,params=get_delayed_gradient_during_trials_simple_diff(exp,cam_ps,prot_ps,parameter_trials,mode,log=log,
                                                                            threshold_strength=threshold_strength,sigma_flow=sigma_flow,tau=tau,
                                                                            min_sequence_length=min_sequence_length,max_strength=max_strength)
        
        deltavs_trials_list.append(deltavs_trials)
        strengthgradients_trials_list.append(strengthgradients_trials)
        params_list.append(params)
        
    return deltavs_trials_list,strengthgradients_trials_list,params_list




def get_gradient_and_flow_speed(params,strengths_thr,fs,vc=None,alpha=None,target=None,tau=0.14,sigma=0.03):

    strengthgradients=[]
    logstrengthgradients=[]
    deltavs=[]
    di=round(tau*fs)
    isigma=round(sigma*fs)
    for i in range(len(params)):
        if vc is not None:
            alphai=params[i]
            vci=vc
        elif alpha is not None:
            vci=params[i] 
            alphai=alpha
        elif target is not None:
            vci=params[i]
            alphai=params[i]/target
        else:
            return
        strength_filt=gaussian_filter(strengths_thr[i],isigma)
        deltavs.append(np.roll(vci-alphai*np.array(strength_filt),di))
        strengthgradients.append(np.gradient(strength_filt,1/fs))    
        logstrengthgradients.append(np.gradient(strength_filt,1/fs)/strength_filt)
        

    return deltavs,strengthgradients,logstrengthgradients

def get_gradient_and_flow_speed_taus(params,strengths_thr,fs,vc,alpha,tau=0.14,sigma=0.03):

    strengthgradients=[]
    logstrengthgradients=[]
    deltavs=[]
    di=round(tau*fs)
    isigma=round(sigma*fs)
    for i in range(len(params)):
        ddi=round(params[i]*fs)
        strength_filt=gaussian_filter(strengths_thr[i],isigma)
        deltavs.append(np.roll(vc-alpha*np.array(strength_filt),di+ddi))
        strengthgradients.append(np.gradient(strength_filt,1/fs))    
        logstrengthgradients.append(np.gradient(strength_filt,1/fs)/strength_filt)
        

    return deltavs,strengthgradients,logstrengthgradients


def calculate_theilsen_slopes_bootstrap(params,deltavs,gradients,thr_vc=0.5,vc=None,n_resamples=100,n_points=1000,pos_quadrant=False):

    slopes_tsbs=np.zeros((len(params),n_resamples))
    for i in range(len(params)):
        if vc is not None:
            mask=(np.abs(deltavs[i])<thr_vc*vc)
        else:
            mask=(np.abs(deltavs[i])<thr_vc*params[i])
        x=deltavs[i][mask]
        y=gradients[i][mask]
               
        if pos_quadrant:
            mask=x*y>0
            x=x[mask]
            y=y[mask]
        
        for n in tqdm(range(n_resamples)):
            random_indices=np.random.randint(len(y),size=n_points)
            slopes_tsbs[i,n]=theilslopes(y[random_indices],x[random_indices])[0]

    slopes_tsbs_mean=np.mean(slopes_tsbs,axis=-1)
    slopes_tsbs_std=np.std(slopes_tsbs,axis=-1)

    return slopes_tsbs_mean,slopes_tsbs_std


def calculate_linear_regression_slopes_bootstrap(params,deltavs,gradients,thr_vc=0.5,vc=None,n_resamples=100,n_points=1000,fit_intercept=False,pos_quadrant=False):

    slopes_lrbs=np.zeros((len(params),n_resamples))
    for i in range(len(params)):
        if vc is not None:
            mask=(np.abs(deltavs[i])<thr_vc*vc)
        else:
            mask=(np.abs(deltavs[i])<thr_vc*params[i])
        x=deltavs[i][mask]
        y=gradients[i][mask]
        
        if pos_quadrant:
            mask=x*y>0
            x=x[mask]
            y=y[mask]
        
        for n in tqdm(range(n_resamples)):
            random_indices=np.random.randint(len(y),size=n_points)
            if fit_intercept:
                slopes_lrbs[i,n]=linregress(x[random_indices],y[random_indices])[0]
            else:
                slopes_lrbs[i,n]=factor_that_minimizes_mse(x[random_indices],y[random_indices])

    slopes_lrbs_mean=np.mean(slopes_lrbs,axis=-1)
    slopes_lrbs_std=np.std(slopes_lrbs,axis=-1)

    return slopes_lrbs_mean,slopes_lrbs_std


def get_frequencies_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05,return_alpha=False,return_target_strength=False,return_v=False,discard_reaction_times=False,
                                        sigma=0.06,window_length=10,min_sequence_length=2,max_strength=10):
    
    frequencies_list=[]
    power_density_list=[]
    amplitude_fit_list=[]
    mean_fit_list=[]
    std_fit_list=[]
    params_list=[]
    alpha_list=[]
    target_strength_list=[]
    v_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        if return_alpha:
            alpha_list.append(prot_ps['alpha_trials'][0])
        if return_target_strength:
            target_strength_list.append(prot_ps['v_flow_trials'][0]/prot_ps['alpha_trials'][0])
        if return_v:
            v_list.append(prot_ps['v_flow_trials'][0])
        parameter_trials=prot_ps[parameter_trials_str]
        
        params,frequencies_trials,power_density_trials,amplitude_fit_trials,mean_fit_trials,std_fit_trials=get_frequencies_during_trials(exp,cam_ps,prot_ps,parameter_trials,threshold_strength=threshold_strength,
                                                     sigma=sigma,window_length=window_length,min_sequence_length=min_sequence_length,max_strength=max_strength)    
            
        params_list.append(params)
        frequencies_list.append(frequencies_trials)
        power_density_list.append(power_density_trials)
        amplitude_fit_list.append(amplitude_fit_trials)
        mean_fit_list.append(mean_fit_trials)
        std_fit_list.append(std_fit_trials)

    
    if return_alpha:
        return params_list,frequencies_list,power_density_list,amplitude_fit_list,mean_fit_list,std_fit_list,alpha_list
    elif return_target_strength:
        return params_list,frequencies_list,power_density_list,amplitude_fit_list,mean_fit_list,std_fit_list,target_strength_list
    elif return_v:
        return params_list,frequencies_list,power_density_list,amplitude_fit_list,mean_fit_list,std_fit_list,v_list
    else:
        return params_list,frequencies_list,power_density_list,amplitude_fit_list,mean_fit_list,std_fit_list,

from scipy.interpolate import interp1d



    
def get_orientation_diffs_during_trials(exp,proj_ps,prot_ps,parameter_trials,time_to_remove=0.01):

    params=np.unique(parameter_trials)
    orientation_diffs=[]
    
    #f_orientation_diff=interp1d(exp['updatetimes'],exp['oris']-exp['v_flow_angle']-exp['oris_pause'],fill_value='extrapolate')
    
    f_orientation_diff=interp1d(exp['updatetimes'],-(exp['oris']-exp['v_flow_angle']-exp['oris_pause'])*np.sign(exp['v_flow_angle']),fill_value='extrapolate')

    for param in params:
        orientation_diffs_param=[]
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]+time_to_remove
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]-time_to_remove
            times=np.linspace(tmin,tmax,round(proj_ps['refresh_rate']*(tmax-tmin)))
            orientation_diffs_param.append(f_orientation_diff(times))
        orientation_diffs.append(orientation_diffs_param)
    
    times=np.linspace(0,tmax-tmin,round(proj_ps['refresh_rate']*(tmax-tmin)))
    
    return orientation_diffs,times,params
    
    
def get_orientation_diffs_multiple_experiments(experiments,parameter_trials_str,time_to_remove=0.01):
    
    orientation_diffs_list=[]
    params_list=[]
    times_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            proj_ps=save_group_as_dict(f,'proj_ps')

        parameter_trials=prot_ps[parameter_trials_str]
        orientation_diffs,times,params=get_orientation_diffs_during_trials(exp,proj_ps,prot_ps,parameter_trials,time_to_remove=time_to_remove)    
            
        orientation_diffs_list.append(orientation_diffs)
        params_list.append(params)
        times_list.append(times)

    return orientation_diffs_list,times_list,params_list
    
    

    
    
    
def get_swimming_fractions_and_thresholded_strengths(strengths,threshold_strength=0.05):
    swimming_fractions=np.zeros(len(strengths))
    strengths_thr=[]
    for i in range(len(strengths)):
        swimming_fractions[i]=np.sum(np.array(strengths[i])>threshold_strength)/len(strengths[i])
        strengths_thr.append(np.array(strengths[i])[np.array(strengths[i])>threshold_strength])
    return strengths_thr,swimming_fractions


def get_signal_histograms(signals,maxv=8,minv=0.03,n_bins=50,scale='lin',density=True,normalize_by_length=False,outside_range=False):
    
    if scale=='lin':
        bin_edges=np.linspace(minv,maxv,n_bins+1-2*int(outside_range))
    elif scale=='log':
        bin_edges=np.logspace(np.log10(minv),np.log10(maxv),n_bins+1-2*int(outside_range))

    if outside_range:
        bin_edges=np.insert(bin_edges,0,-np.inf)
        bin_edges=np.append(bin_edges,np.inf)
        bin_edges_plot=np.copy(bin_edges)
        bin_edges_plot[0]=bin_edges_plot[1]-(bin_edges_plot[2]-bin_edges_plot[1])
        bin_edges_plot[-1]=bin_edges_plot[-2]+(bin_edges_plot[2]-bin_edges_plot[1])
    else:
        bin_edges_plot=np.copy(bin_edges)
    
    densities=np.zeros((len(signals),n_bins))
    for i in range(len(signals)):
        if len(signals[i])>0:
            if normalize_by_length:
                densities[i]=np.histogram(signals[i],bin_edges)[0]/len(signals[i])
            else:
                densities[i]=np.histogram(signals[i],bin_edges,density=density)[0]


    return densities,bin_edges_plot



def get_angular_deflection(exp):
    angular_deflection=np.zeros(len(exp['tail_points_list']))
    angular_deflection=np.rad2deg(np.arctan2(exp['tail_points_list'][:,-1,0],-exp['tail_points_list'][:,-1,1]))                            
    return angular_deflection



def get_strength_densities(parameter_trials,strengths,normalize=True,max_strength=10,n_bins=50):
    params=np.unique(parameter_trials)
    
    #bin_edges=np.linspace(threshold_strength,max_strength,n_bins-1)
    bin_edges=(np.arange(1,n_bins)-0.5)/(n_bins-1)*max_strength

    bin_edges=np.insert(bin_edges,0,-np.inf)
    bin_edges=np.append(bin_edges,np.inf)

    bin_edges_plot=np.copy(bin_edges)

    bin_edges_plot[0]=bin_edges_plot[1]-(bin_edges_plot[2]-bin_edges_plot[1])
    bin_edges_plot[-1]=bin_edges_plot[-2]+(bin_edges_plot[2]-bin_edges_plot[1])
    
    densities=np.zeros((len(params),n_bins))
    for i in range(len(params)):
        if len(strengths[i])>0:
            if normalize:
                densities[i]=np.histogram(strengths[i],bin_edges)[0]/len(strengths[i])
            else:
                densities[i]=np.histogram(strengths[i],bin_edges)[0]
        else:
            densities[i,0]=1
            
    return densities,bin_edges_plot

def get_speed_densities(parameter_trials,strengths,alpha,normalize=True,max_speed=1,n_bins=50):
    params=np.unique(parameter_trials)
    
    #bin_edges=np.linspace(threshold_strength,max_strength,n_bins-1)
    bin_edges=(np.arange(1,n_bins)-0.5)/(n_bins-1)*max_speed

    bin_edges=np.insert(bin_edges,0,-np.inf)
    bin_edges=np.append(bin_edges,np.inf)

    bin_edges_plot=np.copy(bin_edges)

    bin_edges_plot[0]=bin_edges_plot[1]-(bin_edges_plot[2]-bin_edges_plot[1])
    bin_edges_plot[-1]=bin_edges_plot[-2]+(bin_edges_plot[2]-bin_edges_plot[1])
        
    densities=np.zeros((len(params),n_bins))
    for i in range(len(params)):
        if len(strengths[i])>0:
            if normalize:
                densities[i]=np.histogram(alpha*np.array(strengths[i]),bin_edges)[0]/len(strengths[i])
            else:
                densities[i]=np.histogram(alpha*np.array(strengths[i]),bin_edges)[0]
        else:
            densities[i,0]=1
            
    return densities,bin_edges_plot

def get_strength_pdf(parameter_trials,strengths,threshold_strength=0.05,max_strength=10,n_bins=50):
    params=np.unique(parameter_trials)
    
    bin_edges=np.linspace(threshold_strength,max_strength,n_bins)
    #bin_edges=(np.arange(1,n_bins)-0.5)/(n_bins-1)*max_strength

    #bin_edges=np.insert(bin_edges,0,-np.inf)
    bin_edges=np.append(bin_edges,np.inf)

    bin_edges_plot=np.copy(bin_edges)

    #bin_edges_plot[0]=bin_edges_plot[1]-(bin_edges_plot[2]-bin_edges_plot[1])
    bin_edges_plot[-1]=bin_edges_plot[-2]+(bin_edges_plot[2]-bin_edges_plot[1])
    
    densities=np.zeros((len(params),n_bins))
    for i in range(len(params)):
        if len(strengths[i])>0:
            densities[i]=np.histogram(strengths[i],bin_edges,density=True)[0]
        else:
            densities[i,0]=1
            
    return densities,bin_edges_plot

def get_strength_pdf_log(parameter_trials,strengths,threshold_strength=0.05,max_strength=10,n_bins=50):
    params=np.unique(parameter_trials)
    
    bin_edges=np.logspace(np.log10(threshold_strength),np.log10(max_strength),n_bins)
    #bin_edges=(np.arange(1,n_bins)-0.5)/(n_bins-1)*max_strength

    #bin_edges=np.insert(bin_edges,0,-np.inf)
    bin_edges=np.append(bin_edges,np.inf)

    bin_edges_plot=np.copy(bin_edges)

    #bin_edges_plot[0]=bin_edges_plot[1]-(bin_edges_plot[2]-bin_edges_plot[1])
    bin_edges_plot[-1]=bin_edges_plot[-2]*(bin_edges_plot[2]/bin_edges_plot[1])

    densities=np.zeros((len(params),n_bins))
    for i in range(len(params)):
        if len(strengths[i])>0:
            densities[i]=np.histogram(strengths[i],bin_edges,density=True)[0]
        else:
            densities[i,0]=1
            
    return densities,bin_edges_plot

def get_speed_pdf(parameter_trials,v_fb,max_speed=1,n_bins=50):
    params=np.unique(parameter_trials)
    
    bin_edges=np.linspace(0,max_speed,n_bins)
    #bin_edges=(np.arange(1,n_bins)-0.5)/(n_bins-1)*max_strength
    #bin_edges=np.insert(bin_edges,0,-np.inf)
    bin_edges=np.append(bin_edges,np.inf)
    bin_edges_plot=np.copy(bin_edges)
    #bin_edges_plot[0]=bin_edges_plot[1]-(bin_edges_plot[2]-bin_edges_plot[1])
    bin_edges_plot[-1]=bin_edges_plot[-2]+(bin_edges_plot[2]-bin_edges_plot[1])
    
    densities=np.zeros((len(params),n_bins))
    for i in range(len(params)):
        if len(v_fb[i])>0:
            densities[i]=np.histogram(v_fb[i],bin_edges,density=True)[0]
        else:
            densities[i,0]=1
            
    return densities,bin_edges_plot

def get_speed_pdf_log(parameter_trials,v_fb,min_speed=1e-2,max_speed=1,n_bins=50):
    params=np.unique(parameter_trials)
    
    bin_edges=np.logspace(np.log10(min_speed),np.log10(max_speed),n_bins)
    #bin_edges=(np.arange(1,n_bins)-0.5)/(n_bins-1)*max_strength
    #bin_edges=np.insert(bin_edges,0,-np.inf)
    bin_edges=np.append(bin_edges,np.inf)
    bin_edges_plot=np.copy(bin_edges)
    #bin_edges_plot[0]=bin_edges_plot[1]-(bin_edges_plot[2]-bin_edges_plot[1])
    bin_edges_plot[-1]=bin_edges_plot[-2]+(bin_edges_plot[2]-bin_edges_plot[1])
    
    densities=np.zeros((len(params),n_bins))
    for i in range(len(params)):
        if len(v_fb[i])>0:
            densities[i]=np.histogram(v_fb[i],bin_edges,density=True)[0]
        else:
            densities[i,0]=1
            
    return densities,bin_edges_plot





def calculate_bout_durations(exp,prot_ps,cam_ps,threshold_strength=None,):
    if threshold_strength is None:
        threshold_strength=prot_ps['threshold_strength']
    if np.isnan(threshold_strength):
        print('threshold strength is NaN')
        return
    bout_durations=[]
    for trial in range(len(prot_ps['flow_start_times'])):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
        swimming=(exp['strengths'][cameratimes_trial]>threshold_strength).astype(int)
        bout_durations.append(np.array(get_continuous_ones_lengths(swimming))/cam_ps['acquisition_rate'])
    return bout_durations


def get_continuous_ones_lengths(binary_sequence):
    lengths = []
    current_length = 0
    for bit in binary_sequence:
        if bit == 1:
            current_length += 1
        else:
            if current_length > 0:
                lengths.append(current_length)
                current_length = 0
    if current_length > 0:
        lengths.append(current_length)
    return lengths



def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-(x - mean)**2 / (2 * stddev**2))


def gaussian_offset(x, amp, mean, stddev, b):
    return amp * np.exp(-(x - mean)**2 / (2 * stddev**2)) + b


def calculate_and_fit_fourier_spectrum(signal,cam_ps,window_length=10,initial_guess=[10,1,1],acquisition_rate=None,return_spectrum=False):
    if acquisition_rate is None:
        acquisition_rate=cam_ps['acquisition_rate']
    if len(signal)<round(window_length*acquisition_rate):
        print('signal needs to be longer than window length')
        if return_spectrum:        
            return np.nan,np.nan,np.nan,np.nan,np.nan
        else:
            return np.nan,np.nan,np.nan
    else:
        frequencies,power_density=calculate_fourier_spectrum_welch(signal,window_length=window_length,fs=acquisition_rate)

        if return_spectrum:        
            return frequencies,power_density,gaussian_fit_of_fourier_spectrum(frequencies,power_density,initial_guess=initial_guess)
        else:
            return gaussian_fit_of_fourier_spectrum(frequencies,power_density,initial_guess=initial_guess)


def gaussian_fit_of_fourier_spectrum(frequencies,power_density,initial_guess=[10,1,1],offset=False):
    try:
        if offset:
            params, covariance = curve_fit(gaussian_offset,frequencies,power_density,p0=initial_guess+[0,])
            amplitude,mean,std=params
        else:            
            params, covariance = curve_fit(gaussian,frequencies,power_density,p0=initial_guess)
            amplitude,mean,std=params
        if mean<0:
            print('negative frequency?')
            amplitude,mean,std=np.nan,np.nan,np.nan
    except:
        print('fit failed')
        amplitude,mean,std=np.nan,np.nan,np.nan
        
    return amplitude,mean,std




def gaussian_fit_of_fourier_spectrum_strength_change_per_trial(exp,prot_ps,cam_ps,window_length=10,initial_guess=[10,1,1],threshold_strength=0.05,
                                               max_strength=5,sigma_strength=0.06,min_sequence_length=2,relative=False):
    amplitudes=np.zeros(len(prot_ps['flow_start_times']))
    means=np.zeros(len(prot_ps['flow_start_times']))
    stds=np.zeros(len(prot_ps['flow_start_times']))
    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma_strength)
    strengths_change_filt=np.gradient(strengths_filt,1/cam_ps['acquisition_rate'])

    for trial in range(len(amplitudes)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])

        strengths_masked=exp['strengths'][mask]
        if relative:
            strengths_change_filt_masked=strengths_change_filt[mask]*prot_ps['alpha_trials'][trial]/prot_ps['v_flow_trials'][trial]
        else:
            strengths_change_filt_masked=strengths_change_filt[mask]
        sequences=find_consecutive_sequences_mask(strengths_change_filt_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
        sequences=[sequence for sequence in sequences if len(sequence)>=round(min_sequence_length*cam_ps['acquisition_rate'])]
        if len(sequences)>0:
            sequence=np.concatenate(sequences)
            amplitudes[trial],means[trial],stds[trial]=calculate_and_fit_fourier_spectrum(sequence,cam_ps,window_length=window_length,initial_guess=initial_guess)
        else:
            amplitudes[trial],means[trial],stds[trial]=np.nan,np.nan,np.nan

    return amplitudes,means,stds


def gaussian_fit_of_fourier_spectrum_strength_per_trial(exp,prot_ps,cam_ps,window_length=10,initial_guess=[10,1,1],threshold_strength=0.05,
                                               max_strength=5,sigma_strength=0.06,min_sequence_length=2):
    amplitudes=np.zeros(len(prot_ps['flow_start_times']))
    means=np.zeros(len(prot_ps['flow_start_times']))
    stds=np.zeros(len(prot_ps['flow_start_times']))
    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma_strength)

    for trial in range(len(amplitudes)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])

        strengths_masked=exp['strengths'][mask]
        strengths_filt_masked=strengths_filt[mask]
        sequences=find_consecutive_sequences_mask(strengths_filt_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
        sequences=[sequence for sequence in sequences if len(sequence)>=round(min_sequence_length*cam_ps['acquisition_rate'])]
        if len(sequences)>0:
            sequence=np.concatenate(sequences)
            amplitudes[trial],means[trial],stds[trial]=calculate_and_fit_fourier_spectrum(sequence,cam_ps,window_length=window_length,initial_guess=initial_guess)
        else:
            amplitudes[trial],means[trial],stds[trial]=np.nan,np.nan,np.nan

    return amplitudes,means,stds
    
    
    
def get_hilbert_during_trials(exp,cam_ps,prot_ps,parameter_trials,threshold_strength=0.05,max_strength=5,sigma_smooth=0.5,min_sequence_length=2,sigma_strength=0.06):
    

    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma_strength)
    strengths_change_filt=np.gradient(strengths_filt,1/cam_ps['acquisition_rate'])
    params=np.unique(parameter_trials)
    freq=[]
    ampl=[]
    for param in params:
        freq_param=[]
        ampl_param=[]
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])

            strengths_masked=exp['strengths'][mask]
            strengths_change_filt_masked=strengths_change_filt[mask]
            sequences=find_consecutive_sequences_mask(strengths_change_filt_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
            sequences=[sequence for sequence in sequences if len(sequence)>=round(min_sequence_length*cam_ps['acquisition_rate'])]

            if len(sequences)>0:
                sequence=np.concatenate(sequences)
                analytical_signal = hilbert(sequence)
                freqs_hilbert=1/(2*np.pi)*np.gradient(np.unwrap(np.angle(analytical_signal)),1/cam_ps['acquisition_rate'])

                freq_smooth_seq=gaussian_filter(freqs_hilbert,sigma_smooth*cam_ps['acquisition_rate'])
                ampl_smooth_seq=gaussian_filter(np.abs(analytical_signal),sigma_smooth*cam_ps['acquisition_rate'])
                            
                freq_param+=list(freq_smooth_seq)
                ampl_param+=list(ampl_smooth_seq)
        freq.append(freq_param)
        ampl.append(ampl_param)
    return freq,ampl,params


def get_hilbert_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05,max_strength=5,sigma_smooth=0.5,min_sequence_length=2,sigma_strength=0.06):
    
    freq_list=[]
    ampl_list=[]
    params_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        

        parameter_trials=prot_ps[parameter_trials_str]
        freq,ampl,params=get_hilbert_during_trials(exp,cam_ps,prot_ps,parameter_trials,threshold_strength=threshold_strength,max_strength=max_strength,
                                                   sigma_smooth=sigma_smooth,min_sequence_length=min_sequence_length,sigma_strength=sigma_strength)
            
        freq_list.append(freq)
        ampl_list.append(ampl)
        params_list.append(params)
    

    return freq_list,ampl_list,params_list


from scipy.signal import stft,spectrogram



from scipy.signal import find_peaks,peak_widths

def get_spectrum_peak(frequencies,spectrum):
 
    peaks, _=find_peaks(spectrum)
    fwhms,_,_,_=peak_widths(spectrum,peaks)
    fpeak=frequencies[peaks[np.argmax(spectrum[peaks])]]
    fhwhm=fwhms[np.argmax(spectrum[peaks])]*(frequencies[1]-frequencies[0])/2
     
    return fpeak,fhwhm



def get_spectrogram_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05,segment_duration=10,segment_overlap=0.5,max_strength=10,min_sequence_length=5,sigma_strength=0.06,signal_str='sdotovers'):
    
    spectra_list=[]
    f_list=[]
    params_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        

        spectra,f,params=get_spectrogram_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=threshold_strength,
                                                segment_duration=segment_duration,segment_overlap=segment_overlap,
                                                   max_strength=max_strength,min_sequence_length=min_sequence_length,sigma_strength=sigma_strength,signal_str=signal_str)
            
        spectra_list.append(spectra)
        f_list.append(f)
        params_list.append(params)
    

    return spectra_list,f_list,params_list


def get_spectrogram_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=0.05,segment_duration=10,segment_overlap=0.5,
                           max_strength=10,min_sequence_length=5,sigma_strength=0.06,signal_str='sdotovers'):
    
    parameter_trials=prot_ps[parameter_trials_str]

    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma_strength)
    strengths_change_filt=np.gradient(strengths_filt,1/cam_ps['acquisition_rate'])
    
    if signal_str=='sdotovers':
        log_strengths_change_filt=strengths_change_filt/strengths_filt
        signal=log_strengths_change_filt
    elif signal_str=='defl':
        theta_tail=np.zeros(len(exp['tail_points_list']))
        for i in range(len(exp['tail_points_list'])):
            theta_tail[i]=np.rad2deg(np.arctan2(exp['tail_points_list'][i,-1,0],-exp['tail_points_list'][i,-1,1]))
        signal=np.gradient(theta_tail,1/cam_ps['acquisition_rate'])
    else:
        print('signal not understood')
        return
    params=np.unique(parameter_trials)
    spectra=[]
    fs=[]
    for param in params:
        sequences=[]
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])

            strengths_masked=strengths_filt[mask]
            signal_masked=signal[mask]
            consecutive_sequences=find_consecutive_sequences_mask(signal_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
            for seq in consecutive_sequences:
                if len(seq)>=round(min_sequence_length*cam_ps['acquisition_rate']):
                    sequences+=seq
        
        
        #return sequences
    
        if not sequences:
            spectra.append([])
            fs.append([])
        else:
            if len(sequences)>round(segment_duration*cam_ps['acquisition_rate']):
                f, t, Z = spectrogram(np.array(sequences), cam_ps['acquisition_rate'], nperseg=int(segment_duration*cam_ps['acquisition_rate']),noverlap=int(segment_overlap*segment_duration*cam_ps['acquisition_rate']))
                spectra.append(Z)
                fs.append(f)
        
            else:
                spectra.append([])
                fs.append([])

    return spectra,fs,params


def get_stft_multiple_experiments(experiments,parameter_trials_str,threshold_strength=0.05,segment_duration=5,segment_overlap=0.5,max_strength=5,
                                  min_sequence_length=5,sigma_strength=0.06):
    
    spectra_list=[]
    fs_list=[]
    params_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        

        spectra,fs,params=get_stft_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=threshold_strength,
                                                segment_duration=segment_duration,segment_overlap=segment_overlap,
                                                   max_strength=max_strength,min_sequence_length=min_sequence_length,sigma_strength=sigma_strength)
            
        spectra_list.append(spectra)
        fs_list.append(fs)
        params_list.append(params)
    

    return spectra_list,fs_list,params_list

def get_stft_during_trials(exp,cam_ps,prot_ps,parameter_trials_str,threshold_strength=0.05,segment_duration=5,segment_overlap=0.5,
                           max_strength=5,min_sequence_length=5,sigma_strength=0.06):
    
    parameter_trials=prot_ps[parameter_trials_str]

    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma_strength)
    strengths_change_filt=np.gradient(strengths_filt,1/cam_ps['acquisition_rate'])
    params=np.unique(parameter_trials)
    spectra=[]
    for param in params:
        spectra_param=[]
        selected_trials=np.nonzero(parameter_trials==param)[0]

        for trial in selected_trials:
            tmin=prot_ps['flow_start_times'][trial]
            tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
            mask=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])

            strengths_masked=exp['strengths'][mask]
            strengths_change_filt_masked=strengths_change_filt[mask]
            sequences=find_consecutive_sequences_mask(strengths_change_filt_masked,(strengths_masked>=threshold_strength)&(strengths_masked<=max_strength))
            sequences=[sequence for sequence in sequences if len(sequence)>=round(min_sequence_length*cam_ps['acquisition_rate'])]

            if len(sequences)>0:
                sequence=np.concatenate(sequences)
                f, t, Z = spectrogram(sequence, cam_ps['acquisition_rate'], nperseg=int(segment_duration*cam_ps['acquisition_rate']),noverlap=int(segment_overlap*segment_duration*cam_ps['acquisition_rate']))
                
                spectra_param+=[np.abs(Z_i) for Z_i in Z.T]


        spectra.append(spectra_param)
    return spectra,f,params

def get_strength_around_transitions(exp,prot_ps,cam_ps,threshold_strength=0.05,t_before_and_after=2,sigma=0.03,random_fraction=1):

    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma)
    strength_interp_f=interp1d(exp['cameratimes'],strengths_filt,fill_value='extrapolate')

    times_around_transition=np.linspace(-t_before_and_after,t_before_and_after,round(2*t_before_and_after*2*cam_ps['acquisition_rate']))

    strength_trials=[[] for i in range(len(prot_ps['all_transitions']))]
    strength_trial_average=np.zeros((len(prot_ps['all_transitions']),len(times_around_transition)))
    strength_trial_std=np.zeros((len(prot_ps['all_transitions']),len(times_around_transition)))

    for t in range(len(prot_ps['all_transitions'])):
        transition_times=np.copy(prot_ps['transition_times'][t])
        transition_times=transition_times[transition_times!=0] #fix bug where last transition is missing
        strength_all_trials=strength_interp_f(transition_times+times_around_transition[:,np.newaxis])
        for j in range(strength_all_trials.shape[1]):
            if np.all(strength_all_trials[:,j]>threshold_strength):
                strength_trials[t].append(strength_all_trials[:,j])

    for t in range(len(prot_ps['all_transitions'])):
        strength_trials[t]=np.array(strength_trials[t])
        random_indices=np.sort(np.random.choice(np.arange(len(strength_trials[t])),round(len(strength_trials[t])*random_fraction),replace=False))
        strength_trials[t]=strength_trials[t][random_indices]
        strength_trial_average[t]=np.mean(strength_trials[t],axis=0)    
        strength_trial_std[t]=np.std(strength_trials[t],axis=0)  
    
    return strength_trials,strength_trial_average,strength_trial_std,times_around_transition


def get_strength_around_perturbations(exp,prot_ps,cam_ps,threshold_strength=0.05,t_before_and_after=2,sigma=0.03,max_strength=10,perturbation_duration=None):

    if perturbation_duration is None:
        perturbation_duration=prot_ps['perturbation_duration']
        
    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma)
    strength_interp_f=interp1d(exp['cameratimes'],strengths_filt,fill_value='extrapolate')

    times_around_perturbation=np.linspace(-t_before_and_after,perturbation_duration+t_before_and_after,round((2*t_before_and_after+perturbation_duration)*2*cam_ps['acquisition_rate']))

    strength_trials=np.zeros(prot_ps['transition_times'].shape+(len(times_around_perturbation),))
    
    for t in range(prot_ps['transition_times'].shape[0]):
        strength_trials[t]=strength_interp_f(prot_ps['transition_times'][t]+times_around_perturbation[:,np.newaxis]).T
          
    strength_trials_thr=[[] for i in range(prot_ps['transition_times'].shape[0])]
    strength_trial_average=np.zeros((prot_ps['transition_times'].shape[0],len(times_around_perturbation)))
    strength_trial_std=np.zeros((prot_ps['transition_times'].shape[0],len(times_around_perturbation)))

    
    for t in range(prot_ps['transition_times'].shape[0]):
        for j in range(prot_ps['transition_times'].shape[1]):
            strength_during_perturbation=strength_trials[t,j][(times_around_perturbation>0)&(times_around_perturbation<perturbation_duration)]
            #if np.all(strength_during_perturbation>threshold_strength) and np.all(strength_during_perturbation<max_strength):
            if np.any(strength_during_perturbation>threshold_strength):
                strength_trials_thr[t].append(strength_trials[t,j])

        
    for t in range(prot_ps['transition_times'].shape[0]):
        strength_trials_thr[t]=np.array(strength_trials_thr[t])
        #strength_trials_thr[t][strength_trials_thr[t]<threshold_strength]=np.nan
        strength_trial_average[t]=np.nanmean(strength_trials_thr[t],axis=0)
        strength_trial_std[t]=np.nanstd(strength_trials_thr[t],axis=0)  


    
    return strength_trials,strength_trials_thr,strength_trial_average,strength_trial_std,times_around_perturbation


def get_strength_around_perturbations_single_trials(exp,prot_ps,cam_ps,threshold_strength=0.05,t_before_and_after=2,sigma=0.03,max_strength=10):

    strengths_filt=get_strengths_filt(exp,cam_ps,sigma=sigma)
    strength_interp_f=interp1d(exp['cameratimes'],strengths_filt,fill_value='extrapolate')

    times_around_perturbation=np.linspace(-t_before_and_after,prot_ps['perturbation_duration']+t_before_and_after,round((2*t_before_and_after+prot_ps['perturbation_duration'])*2*cam_ps['acquisition_rate']))

    strength_trials=np.zeros((len(prot_ps['perturbation_start_times']),len(times_around_perturbation),))
    strength_trials_ordered=np.zeros((len(prot_ps['vs_alphas']),len(prot_ps['vs_perturbation']),prot_ps['repetitions'],len(times_around_perturbation),))
    
    for t in range(len(prot_ps['perturbation_start_times'])):
        
        strength_trials[t]=strength_interp_f(prot_ps['perturbation_start_times'][t]+times_around_perturbation[:,np.newaxis]).T

    for i in range(len(prot_ps['vs_alphas'])):
        for j in range(len(prot_ps['vs_perturbation'])):

            mask=np.all(prot_ps['vs_alphas'][i]==np.array((prot_ps['v_flow_trials'],prot_ps['alpha_trials'])).T,axis=1)&(prot_ps['vs_perturbation'][j]==prot_ps['v_perturbation_trials'])
            trials=np.nonzero(mask)[0]
            strength_trials_ordered[i,j]=strength_trials[trials]
    
    return strength_trials,strength_trials_ordered,times_around_perturbation


    

def estimate_strength_gradients_after_transition(times_around_transition,strength_trials,prot_ps,set_str,tau=0.14):
    i0=np.argmin(np.abs(times_around_transition-0))
    i1=np.argmin(np.abs(times_around_transition-tau))
    i2=np.argmin(np.abs(times_around_transition-2*tau))
    strength_gradient_trials=[[] for i in range(len(prot_ps['all_transitions']))]
    net_flow_speed_trials=[[] for i in range(len(prot_ps['all_transitions']))]
    fish_speed_trials=[[] for i in range(len(prot_ps['all_transitions']))]
    
    for t in range(len(prot_ps['all_transitions'])):
        for s in range(len(strength_trials[t])):
            strength_gradient_trials[t].append((strength_trials[t][s][i2]-strength_trials[t][s][i1])/(times_around_transition[i2]-times_around_transition[i1]))

            if set_str=='v_set':
                net_flow_speed_trials[t].append(prot_ps['all_transitions'][t][1]-np.mean(prot_ps['alpha']*strength_trials[t][s][i0:i1]))        
                fish_speed_trials[t].append(np.mean(prot_ps['alpha']*strength_trials[t][s][i0:i1]))

            elif set_str=='alpha_set': 
                net_flow_speed_trials[t].append(prot_ps['v']-np.mean(prot_ps['all_transitions'][t][1]*strength_trials[t][s][i0:i1]))        
                fish_speed_trials[t].append(np.mean(prot_ps['all_transitions'][t][1]*strength_trials[t][s][i0:i1]))

        strength_gradient_trials[t]=np.array(strength_gradient_trials[t])
        net_flow_speed_trials[t]=np.array(net_flow_speed_trials[t])
        fish_speed_trials[t]=np.array(fish_speed_trials[t])

    return strength_gradient_trials,net_flow_speed_trials,fish_speed_trials



def autocorrelation_strength_change(exp,cam_ps,t1,t2,threshold_strength=0.05):
    
    strength_filt=get_strengths_filt(exp,cam_ps)
    strength_change_filt=np.gradient(strength_filt,1/cam_ps['acquisition_rate'])
    strength_change_filt_c=strength_change_filt[(exp['strengths']>threshold_strength)&(exp['cameratimes']>t1)&(exp['cameratimes']<t2)]
    autocorr = autocorrelation(strength_change_filt_c)
    lag_times=np.arange(len(autocorr))/cam_ps['acquisition_rate']
    
    return autocorr,lag_times





def crop_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=0.05,max_strength=10,
                                        minimum_region_time=5,time_to_exclude_around_transitions=0.1,tmin=None,tmax=None,return_strength=False,sigma=0.015,return_alpha=False):
    mask=(exp['strengths']>threshold_strength)&(exp['strengths']<max_strength)
    mask_cleaned=set_short_sequences_to_zero(mask,round(cam_ps['acquisition_rate']*minimum_region_time))
    mask_padded=extend_zeros(mask_cleaned,round(cam_ps['acquisition_rate']*time_to_exclude_around_transitions))

    f_mask=interp1d(exp['cameratimes'],mask_padded,kind='nearest',fill_value='extrapolate')
    mask_updatetimes=f_mask(exp['updatetimes']).astype(bool)
    
    if tmin is not None:
        mask_updatetimes[exp['updatetimes']<tmin]=False
    if tmax is not None:
        mask_updatetimes[exp['updatetimes']>tmax]=False

    v_feedback_filt=get_v_feedback_filt(exp,proj_ps)

    updatetimes_c=exp['updatetimes'][mask_updatetimes]
    v_feedback_filt_c=v_feedback_filt[mask_updatetimes]
    v_flow_c=exp['v_flow'][mask_updatetimes]

    list_to_return=[updatetimes_c,v_feedback_filt_c,v_flow_c]                                        

    if return_strength:
        strength_filt=get_strengths_filt(exp,cam_ps,sigma=sigma)
        f_strength_filt=interp1d(exp['cameratimes'],strength_filt,kind='linear',fill_value='extrapolate')
        strength_filt_c=f_strength_filt(updatetimes_c)

        list_to_return.append(strength_filt_c)

    if return_alpha:
        alpha_c=exp['alpha'][mask_updatetimes]
        list_to_return.append(alpha_c)
        
    return list_to_return


def split_by_mask(arr, mask):
    """
    Splits an array into subarrays of consecutive values where the mask is True.

    :param arr: List of values
    :param mask: Boolean list where True indicates the value meets the condition
    :return: List of subarrays with consecutive values where mask is True
    """
    result = []
    subarray = []
    
    for value, condition_met in zip(arr, mask):
        if condition_met:
            subarray.append(value)
        else:
            if subarray:
                result.append(np.array(subarray))
                subarray = []
    if subarray:
        result.append(np.array(subarray))
    
    return result


def select_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=0.05,max_strength=10,
                                        minimum_region_time=5,time_to_exclude_around_transitions=0.1,tmin=None,tmax=None,sigma=None,return_feedback_gain=False):
    
    mask=(exp['strengths']>threshold_strength)&(exp['strengths']<max_strength)
    
    mask_cleaned=set_short_sequences_to_zero(mask,round(cam_ps['acquisition_rate']*minimum_region_time))
    mask_padded=extend_zeros(mask_cleaned,round(cam_ps['acquisition_rate']*time_to_exclude_around_transitions))
    
    f_mask=interp1d(exp['cameratimes'],mask_padded,kind='nearest',fill_value='extrapolate')
    mask_updatetimes=f_mask(exp['updatetimes']).astype(bool)
        
    if tmin is not None:
        mask_updatetimes[exp['updatetimes']<tmin]=False
    if tmax is not None:
        mask_updatetimes[exp['updatetimes']>tmax]=False

    v_feedback_filt=get_v_feedback_filt(exp,proj_ps)
    
    
    v_feedback_filt_c=split_by_mask(v_feedback_filt, mask_updatetimes)
    v_flow_c=split_by_mask(exp['v_flow'], mask_updatetimes)

    if sigma is None:
        strength_filt=exp['strengths']
    else:
        strength_filt=get_strengths_filt(exp,cam_ps,sigma=sigma)
    f_strength_filt=interp1d(exp['cameratimes'],strength_filt,kind='linear',fill_value='extrapolate')
    strength_filt_int=f_strength_filt(exp['updatetimes'])
    
    strength_filt_c=split_by_mask(strength_filt_int, mask_updatetimes)


    if return_feedback_gain:
        gain_c=split_by_mask(exp['alpha'], mask_updatetimes)
        return [v_feedback_filt_c,v_flow_c,strength_filt_c,gain_c]  
        
    return [v_feedback_filt_c,v_flow_c,strength_filt_c]  

def calculate_reverse_correlation(x,y,tw,dt):
    w=round(tw/dt)
    ts=np.arange(w)*dt
    reversecorr=np.zeros(w)
    
    if np.isscalar(x[0]):
        for i in range(len(x)-w):
            reversecorr+=x[i]*y[i:i+w]
        reversecorr/=(len(x)-w)
        
    else:
        n=0
        for l in range(len(x)):
            for i in range(len(x[l])-w):
                reversecorr+=x[l][i]*y[l][i:i+w]
                n+=1
        reversecorr/=n
        
    return ts,reversecorr

def calculate_reverse_correlation_bootstrap(x,y,dt,n_resamples=5,average_block_duration=60,tw=5):

    data=np.array([x,y]).T
    
    reversecorr_bs=[]
    for i in range(n_resamples):
        data_bs=stationary_bootstrap(data,average_block_size=round(average_block_duration/dt))
        
        ts,reversecorr=calculate_reverse_correlation(data_bs[:,0],data_bs[:,1],tw,dt)
        reversecorr_bs.append(reversecorr)
    
    reversecorr_bs=np.array(reversecorr_bs)
    
    reversecorr_mean=np.mean(reversecorr_bs,axis=0)
    reversecorr_std=np.std(reversecorr_bs,axis=0)
        
    return ts,reversecorr_mean,reversecorr_std



def calculate_reverse_correlation_gradient_lists(exp,cam_ps,proj_ps,prot_ps,tw=5,threshold_strength=0.05,max_strength=10,
                                        minimum_region_time=5,time_to_exclude_around_transitions=0.5,tmin=None,tmax=None,sigma=0.03):
    
    
    [v_feedback_filt_c,v_flow_c,strength_filt_c] =select_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=threshold_strength,max_strength=max_strength,
                                        minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,tmin=tmin,tmax=tmax,sigma=sigma)

    xs=[v_flow_c_i-prot_ps['v0'] for v_flow_c_i in v_flow_c]
    ys=[np.gradient(strength_filt_c_i,1/proj_ps['refresh_rate']) for strength_filt_c_i in strength_filt_c]
    
    return calculate_reverse_correlation(xs,ys,tw,1/proj_ps['refresh_rate'])
    
    
def calculate_reverse_correlation_gradient_bootstrap(exp,cam_ps,proj_ps,prot_ps,n_resamples=5,average_block_duration=60,tw=5,threshold_strength=0.05,max_strength=10,
                                        minimum_region_time=5,time_to_exclude_around_transitions=0.5,tmin=None,tmax=None,sigma=0.03):
    
    
    [v_feedback_filt_c,v_flow_c,strength_filt_c]=select_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=threshold_strength,max_strength=max_strength,
                                        minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,tmin=tmin,tmax=tmax,sigma=sigma)

    x=np.concatenate(v_flow_c)-prot_ps['v0'] 
    y=np.gradient(np.concatenate(strength_filt_c),1/proj_ps['refresh_rate']) 
    data=np.array([x,y]).T
    
    reversecorr_bs=[]
    for i in range(n_resamples):
        data_bs=stationary_bootstrap(data,average_block_size=round(average_block_duration*proj_ps['refresh_rate']))
        
        ts,reversecorr=calculate_reverse_correlation(data_bs[:,0],data_bs[:,1],tw,1/proj_ps['refresh_rate'])
        reversecorr_bs.append(reversecorr)
    
    return ts,np.array(reversecorr_bs)


def calculate_reverse_correlation_gradient_bootstrap_estimates(exp,cam_ps,proj_ps,prot_ps,n_resamples=10,average_block_duration=60,tw=5,threshold_strength=0.05,max_strength=10,
                                        minimum_region_time=5,time_to_exclude_around_transitions=0.5,tmin=None,tmax=None,sigma=0.03,log=False):
    
    
    [v_feedback_filt_c,v_flow_c,strength_filt_c]=select_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=threshold_strength,max_strength=max_strength,
                                        minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,tmin=tmin,tmax=tmax,sigma=sigma)

    x=np.concatenate(v_flow_c)-prot_ps['v0'] 
    if log:
        y=np.gradient(np.log(np.concatenate(strength_filt_c)),1/proj_ps['refresh_rate']) 
    else:
        y=np.gradient(np.concatenate(strength_filt_c),1/proj_ps['refresh_rate']) 
    data=np.array([x,y]).T
    
    reversecorr_bs=[]
    for i in range(n_resamples):
        data_bs=stationary_bootstrap(data,average_block_size=round(average_block_duration*proj_ps['refresh_rate']))
        reversecorr_bs.append(calculate_reverse_correlation(data_bs[:,0],data_bs[:,1],tw,1/proj_ps['refresh_rate'])[1])
        
    reversecorr_bs=np.array(reversecorr_bs)    
    
    ts,reversecorr=calculate_reverse_correlation(x,y,tw,1/proj_ps['refresh_rate'])   

    tau_estimate=ts[np.argmax(reversecorr)]
    tau_estimates_bs=ts[np.argmax(reversecorr_bs,axis=1)]

    k_estimate=np.max(np.cumsum(reversecorr/np.std(x)**2))
    k_estimates_bs=np.max(np.cumsum(reversecorr_bs/np.std(x)**2,axis=1),axis=1)

    
    
    return reversecorr,reversecorr_bs,tau_estimate,tau_estimates_bs,k_estimate,k_estimates_bs


def calculate_impulse_response_gradient_bootstrap_estimates(exp,cam_ps,proj_ps,prot_ps,n_resamples=10,average_block_duration=60,tw=5,threshold_strength=0.05,max_strength=10,
                                        minimum_region_time=5,time_to_exclude_around_transitions=0.5,tmin=None,tmax=None,sigma=None,log=False,only_pos=False,only_neg=False,
                                        net_optic_flow=False,response_to_feedback_gain=False):


    if response_to_feedback_gain:
        [v_feedback_filt_c,v_flow_c,strength_filt_c,gain_c]=select_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=threshold_strength,max_strength=max_strength,
                                                    minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,
                                                                                                  tmin=tmin,tmax=tmax,sigma=sigma,return_feedback_gain=True)
        x=np.concatenate(gain_c)-prot_ps['alpha0'] 


    else:
    
        [v_feedback_filt_c,v_flow_c,strength_filt_c]=select_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=threshold_strength,max_strength=max_strength,
                                                        minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,tmin=tmin,tmax=tmax,sigma=sigma)
    
        x=np.concatenate(v_flow_c)-prot_ps['v0'] 

        
    if only_pos:
        x[x<0]=0
    if only_neg:
        x[x>0]=0

    if net_optic_flow:
        x=np.concatenate(v_flow_c)-prot_ps['alpha']*np.concatenate(strength_filt_c)

        

    if log:
        y=np.gradient(np.log(np.concatenate(strength_filt_c)),1/proj_ps['refresh_rate']) 
    else:
        y=np.gradient(np.concatenate(strength_filt_c),1/proj_ps['refresh_rate']) 
    data=np.array([x,y]).T
    
    reversecorr_bs=[]
    for i in range(n_resamples):
        data_bs=stationary_bootstrap(data,average_block_size=round(average_block_duration*proj_ps['refresh_rate']))
        reversecorr_bs.append(calculate_reverse_correlation(data_bs[:,0],data_bs[:,1],tw,1/proj_ps['refresh_rate'])[1])
        
    reversecorr_bs=np.array(reversecorr_bs)    
    
    ts,reversecorr=calculate_reverse_correlation(x,y,tw,1/proj_ps['refresh_rate'])   

    impulse_resp=reversecorr/np.std(x)**2*proj_ps['refresh_rate']
    impulse_resp_bs=reversecorr_bs/np.std(x)**2*proj_ps['refresh_rate']
    
    
    tau_estimate=ts[np.argmax(impulse_resp)]
    tau_estimates_bs=ts[np.argmax(impulse_resp_bs,axis=1)]

    k_estimate=np.max(np.cumsum(impulse_resp/proj_ps['refresh_rate']))
    k_estimates_bs=np.max(np.cumsum(impulse_resp_bs/proj_ps['refresh_rate'],axis=1),axis=1)

    
    
    return impulse_resp,impulse_resp_bs,tau_estimate,tau_estimates_bs,k_estimate,k_estimates_bs,ts


def calculate_impulse_response_gradient_bootstrap_estimates_multiple_experiments(experiments,n_resamples=10,average_block_duration=60,tw=5,threshold_strength=0.05,max_strength=10,minimum_region_time=5,time_to_exclude_around_transitions=0.5,tmin=None,tmax=None,sigma=None,log=False,only_pos=False,only_neg=False,net_optic_flow=False):

    impulse_resp_list=[]
    impulse_resp_m_list=[]
    impulse_resp_s_list=[]
    tau_estimates_list=[]
    tau_estimates_m_list=[]
    tau_estimates_s_list=[]
    k_estimates_list=[]
    k_estimates_m_list=[]
    k_estimates_s_list=[]
    ts_list=[]
    alpha_list=[]
    tau_list=[]
    distance_list=[]


    for experiment in tqdm(experiments):
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            proj_ps=save_group_as_dict(f,'proj_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        impulse_resp,impulse_resp_bs,tau_estimate,tau_estimates_bs,k_estimate,k_estimates_bs,ts=calculate_impulse_response_gradient_bootstrap_estimates(exp,cam_ps,proj_ps,prot_ps,n_resamples=n_resamples,average_block_duration=average_block_duration,tw=tw,threshold_strength=threshold_strength,max_strength=max_strength,minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,tmin=tmin,tmax=tmax,sigma=sigma,log=log,only_pos=only_pos,only_neg=only_neg,net_optic_flow=net_optic_flow)

        if prot_ps['protocol'].decode('utf-8')=='random_velocity_normal':
            distance=0.5
        elif prot_ps['protocol'].decode('utf-8')=='random_velocity_normal_doubled_depth':
            distance=1
        else:
            print('unknown protocol: ',prot_ps['protocol'])

        impulse_resp_list.append(impulse_resp*distance)
        impulse_resp_m_list.append(np.mean(impulse_resp_bs,axis=0)*distance)
        impulse_resp_s_list.append(np.std(impulse_resp_bs,axis=0)*distance)
        tau_estimates_list.append(tau_estimate)
        tau_estimates_m_list.append(np.mean(tau_estimates_bs,axis=0))
        tau_estimates_s_list.append(np.std(tau_estimates_bs,axis=0))
        k_estimates_list.append(k_estimate*distance)
        k_estimates_m_list.append(np.mean(k_estimates_bs,axis=0)*distance)
        k_estimates_s_list.append(np.std(k_estimates_bs,axis=0)*distance)
        ts_list.append(ts)
        alpha_list.append(prot_ps['alpha'])
        tau_list.append(prot_ps['tau'])
        distance_list.append(distance)

    impulse_resp_mean=np.mean(impulse_resp_list,axis=0)
    impulse_resp_std=np.std(impulse_resp_list,axis=0)#/np.sqrt(len(impulse_resp_list))

    return impulse_resp_mean,impulse_resp_std,impulse_resp_list,tau_estimates_list,k_estimates_list,ts_list,alpha_list,tau_list,distance_list,impulse_resp_m_list,impulse_resp_s_list,tau_estimates_m_list,tau_estimates_s_list,k_estimates_m_list,k_estimates_s_list




from sklearn.linear_model import LinearRegression

class AR:
    def __init__(self, p):
        self.p = p
        self.model = LinearRegression()
        self.sigma = None

    def generate_train_x(self, X):
        n = len(X)
        ans = X[:n-self.p]
        ans = np.reshape(ans, (-1, 1))
        for k in range(1, self.p):
            temp = X[k:n-self.p+k]
            temp = np.reshape(temp, (-1, 1))
            ans = np.hstack((ans, temp))
        return ans

    def generate_train_y(self, X):
        return X[self.p:]

    def fit(self, X):
        self.sigma = np.std(X)
        train_x = self.generate_train_x(X)
        train_y = self.generate_train_y(X)
        self.model.fit(train_x, train_y)
        
        
def calculate_timeshifts_sinusoidal_forcing(exp,prot_ps,cam_ps,proj_ps,threshold_strength=0.05,max_strength=10,
                                            minimum_region_time=5,time_to_exclude_around_transitions=0.1,rescale_crosscorr=True):

    timeshifts=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(timeshifts)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration']
        
        updatetimes_c,v_feedback_filt_c,v_flow_c=crop_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=threshold_strength,max_strength=max_strength,
                                        minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,tmin=tmin,tmax=tmax)
        
        if np.sum(updatetimes_c)>0:
            crosscorr,dts=calculate_crosscorr(v_feedback_filt_c-prot_ps['flow_speed'],v_flow_c-prot_ps['flow_speed'],1/proj_ps['refresh_rate'])
            if rescale_crosscorr:
                timeshifts[trial]=dts[np.argmax(crosscorr/(1+dts**2))]
            else:
                timeshifts[trial]=dts[np.argmax(crosscorr)]
            
        else:
            timeshifts[trial]=np.nan
            
    return timeshifts


def calculate_phaseshifts_sinusoidal_forcing_multiple_experiments(experiments,threshold_strength=0.05,max_strength=10,
                                            minimum_region_time=5,time_to_exclude_around_transitions=0.1,rescale_crosscorr=True):

        
    periods_list=[]
    phaseshifts_list=[]
    
    for experiment in experiments:
        path,date,run=experiment

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            proj_ps=save_group_as_dict(f,'proj_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        
        timeshifts=calculate_timeshifts_sinusoidal_forcing(exp,prot_ps,cam_ps,proj_ps,threshold_strength=threshold_strength,max_strength=max_strength,
                                            minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,rescale_crosscorr=rescale_crosscorr)
        
            
        periods_list.append(prot_ps['periods'])
        phaseshifts_list.append(-timeshifts*2*np.pi/prot_ps['periods'])
    

    return periods_list,phaseshifts_list




    timeshifts=np.zeros(len(prot_ps['flow_start_times']))
    for trial in range(len(timeshifts)):
        tmin=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration']
        
        updatetimes_c,v_feedback_filt_c,v_flow_c=crop_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=threshold_strength,max_strength=max_strength,
                                        minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,tmin=tmin,tmax=tmax)
        
        if np.sum(updatetimes_c)>0:
            crosscorr,dts=calculate_crosscorr(v_feedback_filt_c-prot_ps['flow_speed'],v_flow_c-prot_ps['flow_speed'],1/proj_ps['refresh_rate'])
            if rescale_crosscorr:
                timeshifts[trial]=dts[np.argmax(crosscorr/(1+dts**2))]
            else:
                timeshifts[trial]=dts[np.argmax(crosscorr)]
            
        else:
            timeshifts[trial]=np.nan
            
    return timeshifts





def get_k_estimates_regression(path,date,runs,window_width=2,threshold_strength=0.05,max_strength=10,
                                            minimum_region_time=5,time_to_exclude_around_transitions=0.1,sigma=0.03,log_strength=False,square_speed=False):

    k_estimates=[]
    noise_amplitudes=[]    
    
    for run in runs:
        filenames=get_filenames(path,date,run)
        
        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            exp_ps=save_group_as_dict(f,'exp_ps')
            prot_ps=save_group_as_dict(f,'prot_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')
            track_ps=save_group_as_dict(f,'track_ps')
            proj_ps=save_group_as_dict(f,'proj_ps')
            if exp_ps['trigger_fpga']:
                fpga_ps=save_group_as_dict(f,'fpga_ps')
    
    
        updatetimes_c,v_feedback_filt_c,v_flow_c,strength_filt_c=crop_regions_where_fish_is_swimming(exp,cam_ps,proj_ps,threshold_strength=threshold_strength,max_strength=max_strength,
                                                minimum_region_time=minimum_region_time,time_to_exclude_around_transitions=time_to_exclude_around_transitions,
                                                                                                     return_strength=True,sigma=sigma,return_alpha=False)

        
        p=round(window_width*proj_ps['refresh_rate'])
        ar=AR(p=p)
        if square_speed:
            train_x=ar.generate_train_x(raise_to_power(v_flow_c,2))
        else:
            train_x=ar.generate_train_x(v_flow_c)

        if log_strength:
            train_y=ar.generate_train_y(np.gradient(np.log(strength_filt_c),1/proj_ps['refresh_rate']))
        else:
            train_y=ar.generate_train_y(np.gradient((strength_filt_c),1/proj_ps['refresh_rate']))

            
        ar.model.fit(train_x, train_y)
        kernel_current=ar.model.coef_[::-1]*proj_ps['refresh_rate']
        kernel_times=np.arange(p)/proj_ps['refresh_rate']
        k_estimate=np.max(np.cumsum(kernel_current)/proj_ps['refresh_rate'])
        k_estimates.append(k_estimate)
        noise_amplitudes.append(prot_ps['noise_amplitude'])
    
    k_estimates=np.array(k_estimates)
    noise_amplitudes=np.array(noise_amplitudes) 

    return k_estimates,noise_amplitudes