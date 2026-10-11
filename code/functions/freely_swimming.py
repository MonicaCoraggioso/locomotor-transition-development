import os
import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from functions.utilities import *
from functions.visual_flow_behavior_analysis import *
from functions.brain_analysis import *
import pandas as pd
from skimage.filters import threshold_otsu,threshold_li


def print_h5(datapath, **dataframes):
    """Export experimental data"""

    if not os.path.exists(datapath):
        os.makedirs(datapath)

    for name, df in dataframes.items():
        df.to_hdf(os.path.join(datapath, f'{name}.h5'), key='df', mode='w', complevel=9, complib='blosc')
    return


    
def get_onsets_clean(mask_movetail,t_c,mean_thrust_c,mean_thrust_thr=1e-3,t_before_thr=0.06,t_after_thr=0.06,t_before_nan=0.06,t_after_nan=0.5,fraction_thr=0.75):
    mask_movetail_int=mask_movetail.astype(float)
    mask_movetail_int[np.isnan(mean_thrust_c)]=np.nan
    mask_onsets=np.diff(mask_movetail_int)==1
    mask_onsets=np.insert(mask_onsets, 0, False)
    
    onset_indices=np.nonzero(mask_onsets)[0]

    onset_indices_selected=[]
    

    for onset in onset_indices:
        t_onset=t_c[onset]
        indices_before_nan=(t_c>t_onset-t_before_nan)&(t_c<t_onset)
        indices_after_nan=(t_c>t_onset)&(t_c<t_onset+t_after_nan)
        indices_before_thr=(t_c>t_onset-t_before_thr)&(t_c<t_onset)
        indices_after_thr=(t_c>t_onset)&(t_c<t_onset+t_after_thr)
        if np.sum(np.isnan(mean_thrust_c[indices_before_nan|indices_after_nan]))==0 and np.mean(mean_thrust_c[indices_before_thr]<mean_thrust_thr)>fraction_thr and np.mean(mean_thrust_c[indices_after_thr]>mean_thrust_thr)>fraction_thr:
            onset_indices_selected.append(onset)

    return onset_indices_selected

def get_clean_not_movement_segments(mask_movetail, t_c, mean_thrust_c, 
                                       t_before=0.06, t_after=0.06, min_fraction_valid=0.8):
    mask_selected_segments = np.zeros_like(mask_movetail, dtype=bool)

    # Detect ON (rising) transitions
    mask_movetail_int = mask_movetail.astype(float)
    mask_movetail_int[np.isnan(mean_thrust_c)] = np.nan
    mask_onsets = np.diff(mask_movetail_int) == 1
    mask_onsets = np.insert(mask_onsets, 0, False)
    onset_indices = np.nonzero(mask_onsets)[0]

    for onset in onset_indices:
        t_onset = t_c[onset]

        # Check quality of thrust signal around the onset
        indices_window = (t_c > t_onset - t_before) & (t_c < t_onset + t_after)
        segment_thrust = mean_thrust_c[indices_window]

        num_valid = np.sum(~np.isnan(segment_thrust))
        num_total = np.sum(indices_window)
        fraction_valid = num_valid / num_total if num_total > 0 else 0

        if fraction_valid >= min_fraction_valid:
            # Find end of current ON segment
            i = onset
            while i < len(mask_movetail) and mask_movetail[i]:
                i += 1
            mask_selected_segments[onset:i] = True

    return mask_selected_segments


def get_onset_stats(mask_movetail,t_c,mean_thrust_c,onset_indices_selected,vfish_c,vtail_c,t_initial=0.3,quantile_speed=0.9,dt=1/300):
    
    bout_duration=[]
    initial_vfish=[]
    initial_vtail=[]
    
    for onset in onset_indices_selected:
        t_onset=t_c[onset]
        indices_initial=(t_c>t_onset)&(t_c<t_onset+t_initial)
        initial_vfish.append(np.nanquantile(vfish_c[indices_initial],quantile_speed))
        initial_vtail.append(np.nanquantile(vtail_c[indices_initial],quantile_speed))
        bout_duration.append(count_until_zero(mask_movetail,onset)*dt)

    return initial_vfish,initial_vtail,bout_duration



def compute_tail_points_from_pose_angles(thetas,L):
    rotation_matrix = np.array([[0, 1],[-1,  0]])

    tail_points=np.zeros((thetas.shape[0],thetas.shape[1]+1,2))

    for i in tqdm(range(thetas.shape[0])):
        for j in range(thetas.shape[1]):
            tail_points[i,j+1]=tail_points[i,j]+L/thetas.shape[1]*np.array([np.cos(thetas[i,j]),np.sin(thetas[i,j])])

        tail_points[i]=(rotation_matrix@tail_points[i].T).T
    return tail_points


def calculate_mean_thrust_from_pose(tail_points,L,dt=1/300,t_w=0.06,nlast=-1):
    fps=1/dt
    tail_points=tail_points[:,:nlast,:]
    n_points=len(tail_points)
    force=np.zeros((n_points,2))        
    segment_length=L/(n_points-1)
    
    for i in tqdm(range(n_points)):
        segment_centers=(tail_points[min(n_points-1,i+1),1:,:]+tail_points[min(n_points-1,i+1),:-1,:])/2
        segment_centers_d1=(tail_points[i,1:,:]+tail_points[i,:-1,:])/2
        segment_centers_d2=(tail_points[max(0,i-1),1:,:]+tail_points[max(0,i-1),:-1,:])/2
        vel_tail=0.5*fps*(segment_centers-segment_centers_d2)
        segment_directions=np.diff(tail_points[i],axis=0)

        ns_perp=np.transpose((-segment_directions[:,1],segment_directions[:,0])/np.sqrt(segment_directions[:,0]**2+segment_directions[:,1]**2))
        vel_tail_perp=np.sum(vel_tail*ns_perp,axis=-1)
        force_element=-segment_length*np.abs(vel_tail_perp)[:,None]*vel_tail_perp[:,None]*ns_perp

        force[i]=np.sum(force_element,axis=0)
        
        if np.isnan(force[i][1]):
            force[i][1]=0
        if np.isnan(force[i][0]):
            force[i][0]=0
            
            
    w=round(t_w/dt)
    mean_thrust=np.correlate(force[:,1],np.ones(w)/w,mode='same')

    return np.abs(mean_thrust)

    
def get_delay_and_factor_from_exponential_convolution(vfish_sequences,vtail_sequences,taumax=0.2,dt=1/300,ntaus=100):
    
    taus=np.linspace(0,taumax,ntaus)

    best_lmbds=[]
    best_taus=[]
    
    for i in tqdm(range(len(vtail_sequences))):
        vtail_seq=np.array(vtail_sequences[i])
        vfish_seq=np.array(vfish_sequences[i])
        
        lmbd=factor_that_minimizes_mse(vtail_seq,vfish_seq)
        error=np.mean((lmbd*vtail_seq-vfish_seq)**2)
        
        factors=[lmbd]
        errors=[error]
        
        for tau in taus[1:]:
            vtail_conv=convolution_exponential(vtail_seq,dt*np.arange(len(vtail_seq)),tau)
            lmbd=factor_that_minimizes_mse(vtail_conv,vfish_seq)
            factors.append(lmbd)
            error=np.mean((lmbd*vtail_conv-vfish_seq)**2)
            errors.append(error)
    
        best_lmbds.append(factors[np.argmin(errors)])
        best_taus.append(taus[np.argmin(errors)])
    
    best_lmbds=np.array(best_lmbds)
    best_taus=np.array(best_taus)

    return best_lmbds,best_taus

def compute_mask_without_long_quiescence(vtail_c,mask_movetail,max_period_without_tail_movements=10,dt=1/300):
    mask_nomovement=(~mask_movetail)|(np.isnan(vtail_c))
    mask_inactiveperiods=set_short_sequences_to_zero(mask_nomovement,round(max_period_without_tail_movements/dt))
    mask_activeperiods=~mask_inactiveperiods
    return mask_activeperiods


def compute_mask_thrust_thr(mean_thrust_c,mean_thrust_thr=1e-3,pad_time=0.06/4,minimum_sequence_duration=0.06*2,dt=1/300):
    mask_movetail=mean_thrust_c>mean_thrust_thr
    mask_movetail_padded=~extend_zeros(~mask_movetail,round(pad_time/dt))
    mask_movetail_unpadded=extend_zeros(mask_movetail_padded,round(pad_time/dt))
    mask_movetail_c=set_short_sequences_to_zero(mask_movetail_unpadded,round(minimum_sequence_duration/dt))
    return mask_movetail_c

    

def compute_mask_tail_movements(vtail_c,vtail_thr=30,pad_time=0.06/4,minimum_sequence_duration=0.06*2,dt=1/300):
    mask_movetail=vtail_c>vtail_thr
    mask_movetail_padded=~extend_zeros(~mask_movetail,round(pad_time/dt))
    mask_movetail_unpadded=extend_zeros(mask_movetail_padded,round(pad_time/dt))
    mask_movetail_c=set_short_sequences_to_zero(mask_movetail_unpadded,round(minimum_sequence_duration/dt))
    return mask_movetail_c
    
def compute_hilbert(y,dt):
    analytic_signal = hilbert(y)
    ampl_h=np.abs(analytic_signal)
    phase_h=np.unwrap(np.angle(analytic_signal))
    freq_h=np.gradient(phase_h,dt)/(2*np.pi)
    return ampl_h,freq_h


def clean_data_and_interpolate_with_thrust(tfish,vfish,ttail,angletail,mean_thrust,L,dt=1/300,maximum_distance_from_data=0.015,time_to_remove_around_missing_data=0.2,minimum_sequence_duration=0.5,kind='linear'):
    t_c=np.arange(int(np.max(np.concatenate([tfish,ttail]))/dt)+1)*dt
    vfish_int=remove_missing_data_and_interpolate(tfish,vfish,t_c,
                                                  maximum_distance_from_data=maximum_distance_from_data,time_to_remove_around_missing_data=time_to_remove_around_missing_data,
                                                  minimum_sequence_duration=minimum_sequence_duration,kind=kind)
    angletail_int=remove_missing_data_and_interpolate(ttail,angletail,t_c,
                                                  maximum_distance_from_data=maximum_distance_from_data,time_to_remove_around_missing_data=time_to_remove_around_missing_data,
                                                  minimum_sequence_duration=minimum_sequence_duration,kind=kind)
    mean_thrust_int=remove_missing_data_and_interpolate(ttail,mean_thrust,t_c,
                                                  maximum_distance_from_data=maximum_distance_from_data,time_to_remove_around_missing_data=time_to_remove_around_missing_data,
                                                  minimum_sequence_duration=minimum_sequence_duration,kind=kind)
    dangletail_int=np.gradient(angletail_int,t_c)
    mask_missing_either=np.isnan(vfish_int)|np.isnan(dangletail_int)
    mask_missing_either_thr=~set_short_sequences_to_zero(~mask_missing_either,round(minimum_sequence_duration/dt))
    vfish_c=np.copy(vfish_int)
    vfish_c[mask_missing_either_thr]=np.nan
    angletail_c=np.copy(angletail_int)
    angletail_c[mask_missing_either_thr]=np.nan
    dangletail_c=np.copy(dangletail_int)
    dangletail_c[mask_missing_either_thr]=np.nan
    mean_thrust_c=np.copy(mean_thrust_int)
    mean_thrust_c[mask_missing_either_thr]=np.nan
    
    vtail_c,freqtail_c=compute_hilbert_sequences_not_nan(L*dangletail_c,dt)

    return t_c,vfish_c,vtail_c,dangletail_c,angletail_c,freqtail_c,mean_thrust_c
    

def clean_data_and_interpolate(tfish,vfish,ttail,angletail,L,dt=1/300):
    t_c=np.arange(int(np.max(np.concatenate([tfish,ttail]))/dt)+1)*dt
    vfish_int=remove_missing_data_and_interpolate(tfish,vfish,t_c)
    angletail_int=remove_missing_data_and_interpolate(ttail,angletail,t_c)
    dangletail_int=np.gradient(angletail_int,t_c)
    _,angletail_c=remove_missing_data_either_fish_or_tail(vfish_int,angletail_int,t_c)
    vfish_c,dangletail_c=remove_missing_data_either_fish_or_tail(vfish_int,dangletail_int,t_c)
    vtail_c,freqtail_c=compute_hilbert_sequences_not_nan(L*dangletail_c,dt)

    return t_c,vfish_c,vtail_c,dangletail_c,angletail_c,freqtail_c

def compute_hilbert_sequences_not_nan(y,dt):
    y_sequences=find_consecutive_sequences_mask(y,~np.isnan(y))
    ampls=[]
    freqs=[]
    for y_seq in y_sequences:
        analytic_signal = hilbert(y_seq)
        ampls.append(np.abs(analytic_signal))
        freqs.append(np.gradient(np.unwrap(np.angle(analytic_signal)),dt)/(2*np.pi))

    ampl_h=np.copy(y)
    ampl_h[~np.isnan(y)]=np.concatenate(ampls)
    freq_h=np.copy(y)
    freq_h[~np.isnan(y)]=np.concatenate(freqs)

    return ampl_h,freq_h

def load_h5(path, *names, verbose=True):
    """
    Load multiple pandas DataFrames from .h5 files by name.
    
    Parameters:
    path (str): Directory where the .h5 files are stored.
    *names (str): Names of the DataFrames to load.
    verbose (bool): If True, prints confirmation for each loaded file.
    
    Returns:
    list: Loaded DataFrames in the same order as names.
    """
    dfs = []
    for name in names:
        filepath = os.path.join(path, f"{name}.h5")
        df = pd.read_hdf(filepath, key='df')
        dfs.append(df)
        if verbose:
            print(f"📥 Loaded {name} ← {filepath}")
    return dfs



def subset_by_fish(fish_ID, **dfs):
    """
    Subsets each provided DataFrame to include only rows where the 'ID' column starts with the given fish_ID.

    Parameters:
        fish_ID (str): The fish identifier prefix to match (e.g., '14_31Oct24_1').
        **dfs: Arbitrary keyword arguments where each value is a DataFrame containing an 'ID' column.
               The function returns a subset of each DataFrame where 'ID'.startswith(fish_ID) is True.

    Returns:
        tuple: A tuple of subset DataFrames (one for each input DataFrame), in the same order as
               the keyword arguments were passed.

    Example:
        DF1, DF2 = subset_by_fish('14_31Oct24_1', DF1=df1, DF2=df2)
        # DF1 will include rows where df1['ID'].startswith('14_31Oct24_1')
    """

    print(f'Subsetting data of the fish {fish_ID:s}')

    results = []
    for name, df in dfs.items():
        mask = df['ID'].str.startswith(fish_ID).reset_index(drop=True)
        results.append(df[mask])
    return tuple(results)


def remove_missing_data_and_interpolate(t,y,new_t,maximum_distance_from_data=0.015,time_to_remove_around_missing_data=0.2,minimum_sequence_duration=0.5,kind='linear'):
    t = np.asarray(t)
    y = np.asarray(y)
    new_t = np.asarray(new_t)

    interpolator = interp1d(t, y, kind=kind, bounds_error=False, fill_value=np.nan)

    tree = cKDTree(t[:, None])
    distances, _ = tree.query(new_t[:, None], k=1)

    y_int = interpolator(new_t)
    
    mask_missing=distances>maximum_distance_from_data
    dt=np.mean(np.diff(new_t))
    mask_missing_padded=~extend_zeros(~mask_missing,round(time_to_remove_around_missing_data/dt))
    mask_missing_thr=~set_short_sequences_to_zero(~mask_missing_padded,round(minimum_sequence_duration/dt))

    y_int[mask_missing_thr]=np.nan

    return y_int


def remove_missing_data_either_fish_or_tail(vfish_int,vtail_int,t_int,minimum_sequence_duration=0.5):
    mask_missing_either=np.isnan(vfish_int)|np.isnan(vtail_int)
    dt=np.mean(np.diff(t_int))
    mask_missing_either_thr=~set_short_sequences_to_zero(~mask_missing_either,round(minimum_sequence_duration/dt))
    vfish_int_c=np.copy(vfish_int)
    vfish_int_c[mask_missing_either_thr]=np.nan
    vtail_int_c=np.copy(vtail_int)
    vtail_int_c[mask_missing_either_thr]=np.nan
    return vfish_int_c,vtail_int_c
    

def thresholded_interp(t, y, new_t, threshold, minimum_duration=0,remove_border=0,kind='linear'):
    t = np.asarray(t)
    y = np.asarray(y)
    new_t = np.asarray(new_t)

    interpolator = interp1d(t, y, kind=kind, bounds_error=False, fill_value=np.nan)

    tree = cKDTree(t[:, None])
    distances, _ = tree.query(new_t[:, None], k=1)

    result = interpolator(new_t)

    mask=distances<threshold

    dt=np.mean(np.diff(new_t))
    mask_padded=extend_zeros(mask,round(remove_border/dt))
    mask_cleaned=set_short_sequences_to_zero(mask_padded,round(minimum_duration/dt))

    result[~mask_cleaned] = np.nan

    return result




def calculate_speed_lighthill(tail_points,tau_lp=0,n=1,fish_velocity=0,fps=250):
    n_points=len(tail_points)
    force=np.zeros((n_points,2))
    v=np.zeros(n_points)
    v_side=np.zeros(n_points)   
    torque=np.zeros(n_points)
    omega=np.zeros(n_points)
    if tau_lp==0:
        beta_lp=0
    else:
        beta_lp=np.exp(-1/(fps*tau_lp))
    if type(fish_velocity) == int or float:
        fish_velocity=fish_velocity*np.ones(n_points)


    for i in range(n_points):

        segment_centers=(tail_points[min(n_points-1,i+1),1:,:]+tail_points[min(n_points-1,i+1),:-1,:])/2
        segment_centers_d1=(tail_points[i,1:,:]+tail_points[i,:-1,:])/2
        segment_centers_d2=(tail_points[max(0,i-1),1:,:]+tail_points[max(0,i-1),:-1,:])/2
        vel_tail_tip=0.5*fps*(segment_centers[-1]-segment_centers_d2[-1])+(0,fish_velocity[i])
        tip_direction=tail_points[i,-1]-tail_points[i,-2]
        ns_perp=np.transpose((-tip_direction[1],tip_direction[0])/np.sqrt(tip_direction[0]**2+tip_direction[1]**2))
        ns_par=np.transpose((-tip_direction[0],-tip_direction[1])/np.sqrt(tip_direction[0]**2+tip_direction[1]**2))
        vel_perp=np.sum(vel_tail_tip*ns_perp)

        force[i]=(-vel_perp*vel_tail_tip[1]-0.5*vel_perp**2*ns_par[0],vel_perp*vel_tail_tip[0]-0.5*vel_perp**2*ns_par[1])
        force_exp=np.sign(force[i])*np.abs(force[i])**n

        torque[i]=np.cross(segment_centers_d1[-1],force[i])
        torque_exp=np.sign(torque[i])*np.abs(torque[i])**n

        if np.isnan(force_exp[0]):
            force_exp[0]=0
        if np.isnan(force_exp[1]):
            force_exp[1]=0
        if np.isnan(torque_exp):
            torque_exp=0
            
        v[i]=v[i-1]*beta_lp+force_exp[1]*(1-beta_lp)
        v_side[i]=v_side[i-1]*beta_lp+force_exp[0]*(1-beta_lp)
        omega[i]=omega[i-1]*beta_lp+torque_exp*(1-beta_lp)
    return force,v,v_side,torque,omega

def calculate_speed_form(tail_points,segment_length,tau_lp=0,n=1,fish_velocity=0,fps=250):
    n_points=len(tail_points)
    force=np.zeros((n_points,2))
    v=np.zeros(n_points)
    vl=np.zeros(n_points)
    torque=np.zeros(n_points)
    omega=np.zeros(n_points)
    if tau_lp==0:
        beta_lp=0
    else:
        beta_lp=np.exp(-1/(fps*tau_lp))
        
    if type(fish_velocity) == int or float:
        fish_velocity=fish_velocity*np.ones(n_points)
        
    for i in range(n_points):
        segment_centers=(tail_points[min(n_points-1,i+1),1:,:]+tail_points[min(n_points-1,i+1),:-1,:])/2
        segment_centers_d1=(tail_points[i,1:,:]+tail_points[i,:-1,:])/2
        segment_centers_d2=(tail_points[max(0,i-1),1:,:]+tail_points[max(0,i-1),:-1,:])/2
        vel_tail=0.5*fps*(segment_centers-segment_centers_d2)+(0,fish_velocity[i])
        segment_directions=np.diff(tail_points[i],axis=0)


        ns_perp=np.transpose((-segment_directions[:,1],segment_directions[:,0])/np.sqrt(segment_directions[:,0]**2+segment_directions[:,1]**2))

        vel_tail_perp=np.sum(vel_tail*ns_perp,axis=-1)

        force_element=-segment_length*np.abs(vel_tail_perp)[:,None]*vel_tail_perp[:,None]*ns_perp

        force[i]=np.sum(force_element,axis=0)
        force[i]=np.sign(force[i])*np.abs(force[i])**n

        torque[i]=np.sum(np.cross(segment_centers_d1,force_element),axis=0)
        torque[i]=np.sign(torque[i])*np.abs(torque[i])**n
        
        if np.isnan(force[i][1]):
            force[i][1]=0
        if np.isnan(force[i][0]):
            force[i][0]=0
        if np.isnan(torque[i]):
            torque[i]=0
        v[i]=v[i-1]*beta_lp+force[i][1]*(1-beta_lp)
        vl[i]=vl[i-1]*beta_lp+force[i][0]*(1-beta_lp)
        omega[i]=omega[i-1]*beta_lp+torque[i]*(1-beta_lp)


    return force,v,vl,torque,omega


def time_difference(signal,deltat,dt):
    di=round(deltat/dt)
    return signal[di:]-signal[:-di]

def integrate(signal,dt):
    return dt*np.cumsum(signal)

def factor_that_minimizes_mse(signal,target):
    return np.nanmean(signal*target)/np.nanmean(signal**2)

def minimal_rmse(signal,target):
    A=factor_that_minimizes_mse(signal,target)
    return np.mean((target-A*signal)**2)**0.5

def minimal_r2(signal,target):
    A=factor_that_minimizes_mse(signal,target)
    return 1-np.mean((target-A*signal)**2)/np.mean((target-np.mean(target))**2)
    
def minimal_rmse_with_offset(signal,target):
    A,B=factor_and_offset_that_minimize_mse(signal,target)
    return np.mean((target-A*signal-B)**2)**0.5

def factor_and_offset_that_minimize_mse(signal,target):
    A=(np.mean(signal*target)-np.mean(signal)*np.mean(target))/(np.mean(signal**2)-np.mean(signal)**2)
    B=np.mean(target-A*signal)
    return A,B

def integrate_nonlinear_equation(signal,gamma,g,dt):
    signal_integrated=np.zeros_like(signal)
    for i in range(len(signal)):
        signal_integrated[i]=signal_integrated[i-1]+(-signal_integrated[i-1]**gamma+g*signal[i-1])*dt
    return signal_integrated

def integrate_nonlinear_equation_square(signal,lmbd,g,dt):
    signal_integrated=np.zeros_like(signal)
    for i in range(len(signal)):
        signal_integrated[i]=signal_integrated[i-1]+(-signal_integrated[i-1]**2/lmbd+g*signal[i-1])*dt
    return signal_integrated

    
def lowpass(signal,tau,dt):
    if tau==0:
        return signal
    else:
        signal_lp=np.zeros_like(signal)
        beta_lp=np.exp(-dt/tau)
        for i in range(len(signal)):
            signal_lp[i]=signal_lp[i-1]*beta_lp+signal[i]*(1-beta_lp)
        return signal_lp
    
    
    