import numpy as np
import h5py
from ipywidgets import interact
import matplotlib.pyplot as plt
from functions.visual_flow_behavior_analysis import *
from functions.utilities import *

def plot_stimulus_images(filenames,t0=0):
    '''
    plot visual stimuli with a slider
    '''
    with h5py.File(filenames['stimulus_pictures'],'r') as f:
        n_frames=len(f['savetimes'])
        frame0=np.argmin(np.abs(f['savetimes'][:]-t0))
    @interact(frame=(0,n_frames-1,1))
    def _plot_stimulus_images(frame=frame0):
        with h5py.File(filenames['stimulus_pictures'],'r') as f:
            plt.close()
            fig=plt.figure()
            im=plt.imshow(f['stimulus_pictures'][frame],cmap='gray')
            plt.colorbar()
            print(np.round(f['savetimes'][frame],3))
            plt.show()
    

def plot_tail_images(filenames,exp,cam_ps,track_ps,t0=0):
    '''
    plot segmented tail images with a slider
    '''
    n_frames=len(exp['cameratimes'])
    frame0=np.argmin(np.abs(exp['cameratimes']-t0))
    @interact(frame=(0,n_frames-1,1))
    def _plot_tail_images(frame=frame0):
        with h5py.File(filenames['tail_pictures'],'r') as f:
            plt.close()
            fig=plt.figure()
            im=plt.imshow(f['tail_pictures'][frame],cmap='gray')
            plt.colorbar()
            tail_points_retransformed=retransform_tail_points(exp['tail_points_list'][frame],cam_ps,track_ps)
            line,=plt.plot(tail_points_retransformed[:,0],tail_points_retransformed[:,1],'r.',alpha=0.5)
            print(np.round(exp['cameratimes'][frame],3))
            plt.show()


def plot_tail_images_rotated(filenames,exp,cam_ps,track_ps,t0=0):
    '''
    plot rotated segmented tail images with a slider
    '''
    n_frames=len(exp['cameratimes'])
    tail_points_retransformed=np.zeros_like(exp['tail_points_list'][0])
    frame0=np.argmin(np.abs(exp['cameratimes']-t0))
    @interact(frame=(0,n_frames-1,1))
    def _plot_tail_images(frame=frame0):
        with h5py.File(filenames['tail_pictures'],'r') as f:
            plt.close()
            fig=plt.figure()
            im=plt.imshow(f['tail_pictures'][frame],cmap='gray')
            plt.colorbar()
            for j in range(len(tail_points_retransformed)):
                tail_points_retransformed[j]=-exp['tail_points_list'][frame,j][[1,0]]
                tail_points_retransformed[j]=np.matmul(track_ps['R'].T,tail_points_retransformed[j]/cam_ps['cm_per_pix'])+track_ps['initial_pos'] 
            line,=plt.plot(tail_points_retransformed[:,0],tail_points_retransformed[:,1],'r.',alpha=0.5)
            print(np.round(exp['cameratimes'][frame],3))
            plt.show()

            
def plot_tail_points(exp,t0=0):
    '''
    plot segmented tail points with a slider
    '''
    n_frames=len(exp['cameratimes'])
    frame0=np.argmin(np.abs(exp['cameratimes']-t0))
    xmin=np.nanmin(exp['tail_points_list'][:,:,0])
    xmax=np.nanmax(exp['tail_points_list'][:,:,0])
    ymin=np.nanmin(exp['tail_points_list'][:,:,1])
    ymax=np.nanmax(exp['tail_points_list'][:,:,1])
    xmin-=0.05*(xmax-xmin)
    xmax+=0.05*(xmax-xmin)
    ymin-=0.05*(ymax-ymin)
    ymax+=0.05*(ymax-ymin)
    @interact(frame=(0,n_frames-1,1))
    def _plot_tail_points(frame=frame0):
        plt.close()
        fig,ax=plt.subplots()
        ax.plot(exp['tail_points_list'][frame,:,0],exp['tail_points_list'][frame,:,1],'k.-')
        ax.set_aspect('equal')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        print(np.round(exp['cameratimes'][frame],3))
        plt.show()
        
        
def plot_tail_points_velocity(exp,cam_ps,t0=0,scale=1):
    '''
    plot segmented tail points with arrows indicating tail speed with a slider
    '''
    n_frames=len(exp['cameratimes'])
    frame0=np.argmin(np.abs(exp['cameratimes']-t0))
    xmin=np.nanmin(exp['tail_points_list'][:,:,0])
    xmax=np.nanmax(exp['tail_points_list'][:,:,0])
    ymin=np.nanmin(exp['tail_points_list'][:,:,1])
    ymax=np.nanmax(exp['tail_points_list'][:,:,1])
    xmin-=0.05*(xmax-xmin)
    xmax+=0.05*(xmax-xmin)
    ymin-=0.2*(ymax-ymin)
    ymax+=0.05*(ymax-ymin)
    @interact(frame=(0,n_frames-2,1))
    def _plot_tail_points(frame=frame0):
        i=frame
        segment_centers=(exp['tail_points_list'][i+1,1:,:]+exp['tail_points_list'][i+1,:-1,:])/2
        segment_centers_d1=(exp['tail_points_list'][i,1:,:]+exp['tail_points_list'][i,:-1,:])/2
        segment_centers_d2=(exp['tail_points_list'][i-1,1:,:]+exp['tail_points_list'][i-1,:-1,:])/2
        vel_tail_tip=0.5*cam_ps['acquisition_rate']*(segment_centers[-1]-segment_centers_d2[-1])
        tip_direction=exp['tail_points_list'][i,-1]-exp['tail_points_list'][i,-2]
        ns_perp=np.transpose((-tip_direction[1],tip_direction[0])/np.sqrt(tip_direction[0]**2+tip_direction[1]**2))
        ns_par=np.transpose((-tip_direction[0],-tip_direction[1])/np.sqrt(tip_direction[0]**2+tip_direction[1]**2))
        vel_perp=np.sum(vel_tail_tip*ns_perp)
        vel_par=np.sum(vel_tail_tip*ns_par)

        vel_perp_vec=vel_perp*ns_perp
        vel_par_vec=vel_par*ns_par
        
        plt.close()
        fig,ax=plt.subplots()
        ax.plot(exp['tail_points_list'][frame,:,0],exp['tail_points_list'][frame,:,1],'k.-')

        ax.arrow(segment_centers_d1[-1][0],segment_centers_d1[-1][1],vel_perp_vec[0]*scale,vel_perp_vec[1]*scale,color='r')
        ax.arrow(segment_centers_d1[-1][0],segment_centers_d1[-1][1],vel_par_vec[0]*scale,vel_par_vec[1]*scale,color='b')
        
        ax.set_aspect('equal')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        print(np.round(exp['cameratimes'][frame],3))
        plt.show()
                
        

def plot_trials(exp,prot_ps,proj_ps,sigma_strength=0.025,sigma_turning_strength=0.05):
    '''
    plot behavioral signals trial by trial with a slider
    '''
    n_trials=len(prot_ps['pause_start_times'])
    v_flow_angle_nan=np.copy(exp['v_flow_angle'])
    v_flow_angle_nan[exp['v_flow']==0]=np.nan
    v_feedback_filt=gaussian_filter(exp['v_feedback'],sigma_strength*proj_ps['refresh_rate'])
    omega_feedback_filt=gaussian_filter(exp['omega_feedback'],sigma_turning_strength*proj_ps['refresh_rate'])
    
    @interact(trial=(0,n_trials-1,1))
    def _plot_trials(trial=0):
        plt.close()
        fig,axs=plt.subplots(2,3,figsize=(6.4*2,4.8*2),dpi=75,sharex=True)
        print('v_flow='+str(prot_ps['v_flow_trials'][trial])+', spatial_period='+str(prot_ps['spatial_period_trials'][trial])+', alpha='+str(prot_ps['alpha_trials'][trial])+
                     ', beta='+str(prot_ps['beta_trials'][trial])+', v_flow_angle='+str(prot_ps['v_flow_angle_trials'][trial])+', omega_flow='+str(prot_ps['omega_flow_trials'][trial])+
                     ', metric='+str(prot_ps['metric_trials'][trial])+', tau_lp='+str(prot_ps['tau_lp_trials'][trial])+', texture_index='+str(prot_ps['texture_index_trials'][trial])+
                     ', threshold_strength='+str(prot_ps['threshold_strength_trials'][trial]))
        if 'tau_trials' in prot_ps:
            print('tau='+str(prot_ps['tau_trials'][trial]))
                    
            
            
        axs_r=axs.ravel()
        tmin=prot_ps['pause_start_times'][trial]
        thalf=prot_ps['flow_start_times'][trial]
        tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
        cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
        fliptimes_trial=tuple([(exp['fliptimes']>tmin)&(exp['fliptimes']<tmax)])
        fliptimes_trial_flow=tuple([(exp['fliptimes']>thalf)&(exp['fliptimes']<tmax)])
        ax=axs_r[0]
        ax.plot(exp['cameratimes'][cameratimes_trial],exp['deflections'][cameratimes_trial],lw=1)
        ax.set_xlim(tmin,tmax)
        ax.set_ylabel('deflection')
        ax=axs_r[1]
        ax.plot(exp['fliptimes'][fliptimes_trial],v_feedback_filt[fliptimes_trial])
        ax.plot(exp['fliptimes'][fliptimes_trial],exp['v_flow'][fliptimes_trial])
        ax.set_xlim(tmin,tmax)
        ax.set_ylabel('speed')
        ax=axs_r[2]
        ax.plot(exp['fliptimes'][fliptimes_trial],omega_feedback_filt[fliptimes_trial])
        ax.plot(exp['fliptimes'][fliptimes_trial],exp['omega_flow'][fliptimes_trial])
        ax.set_xlim(tmin,tmax)
        ax.set_ylabel('angular velocity')
        ax=axs_r[3]
        ax.plot(exp['fliptimes'][fliptimes_trial_flow],prot_ps['spatial_period_trials'][trial]*(np.linalg.norm(exp['phases'][fliptimes_trial_flow]-exp['phases'][fliptimes_trial_flow][0],axis=-1)))
        ax.set_xlim(tmin,tmax)
        ax.set_ylabel('displacement')
        ax=axs_r[4]
        ax.plot(exp['fliptimes'][fliptimes_trial],exp['oris'][fliptimes_trial])
        ax.plot(exp['fliptimes'][fliptimes_trial],v_flow_angle_nan[fliptimes_trial]+exp['oris_pause'][fliptimes_trial])
        ax.set_xlim(tmin,tmax)
        ax.set_ylabel('angle')
        plt.show()

        

def plot_trials_experiments_speed(experiments,parameter_trials_str,e0=0,p0=0):

    @interact(e=(0,len(experiments)-1,1))
    def plot1(e=e0):

        path,date,run=experiments[e]

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            proj_ps=save_group_as_dict(f,'proj_ps')

        parameter_trials=prot_ps[parameter_trials_str]
        params=np.unique(parameter_trials)

        @interact(p=(0,len(params)-1,1))
        def plot2(p=p0):

            param=params[p]
            print(param)

            v_feedback_filt=get_v_feedback_filt(exp,proj_ps)
            selected_trials=np.nonzero(parameter_trials==param)[0]

            plt.close('all')
            for trial in selected_trials:
                fig,ax=plt.subplots(dpi=75)

                tmin=prot_ps['pause_start_times'][trial]
                thalf=prot_ps['flow_start_times'][trial]
                tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
                cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
                fliptimes_trial=tuple([(exp['fliptimes']>tmin)&(exp['fliptimes']<tmax)])
                ax.plot(exp['fliptimes'][fliptimes_trial],v_feedback_filt[fliptimes_trial])
                ax.plot(exp['fliptimes'][fliptimes_trial],exp['v_flow'][fliptimes_trial])
                ax.set_xlim(tmin,tmax)
                plt.show()

                
def plot_trials_experiments_speed_sinusoidal_forcing(experiments,e0=0,p0=0):

    @interact(e=(0,len(experiments)-1,1))
    def plot1(e=e0):

        path,date,run=experiments[e]

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            proj_ps=save_group_as_dict(f,'proj_ps')

        parameter_trials=prot_ps['periods']
        params=np.unique(parameter_trials)

        @interact(p=(0,len(params)-1,1))
        def plot2(p=p0):

            param=params[p]
            print(param)

            v_feedback_filt=get_v_feedback_filt(exp,proj_ps)
            selected_trials=np.nonzero(parameter_trials==param)[0]

            plt.close('all')
            for trial in selected_trials:
                fig,ax=plt.subplots(dpi=75)

                tmin=prot_ps['pause_start_times'][trial]
                thalf=prot_ps['flow_start_times'][trial]
                tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration']
                cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
                fliptimes_trial=tuple([(exp['fliptimes']>tmin)&(exp['fliptimes']<tmax)])
                ax.plot(exp['fliptimes'][fliptimes_trial],v_feedback_filt[fliptimes_trial])
                ax.plot(exp['fliptimes'][fliptimes_trial],exp['v_flow'][fliptimes_trial])
                ax.set_xlim(tmin,tmax)
                plt.show()               


def mean_rolling_filter(signal, window_size):
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')


def plot_trials_experiments_strength(experiments,parameter_trials_str,e0=0,p0=0,ymax=None,ymin=0,log=False,window_duration=None):

    @interact(e=(0,len(experiments)-1,1))
    def plot1(e=e0):

        path,date,run=experiments[e]

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            proj_ps=save_group_as_dict(f,'proj_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        parameter_trials=prot_ps[parameter_trials_str]
        params=np.unique(parameter_trials)

        @interact(p=(0,len(params)-1,1))
        def plot2(p=p0):

            param=params[p]
            print(param)
    
            if window_duration is None:
                strengths_filt=get_strengths_filt(exp,cam_ps)
            else:
                window_size=round(window_duration*cam_ps['acquisition_rate'])
                strengths_filt=mean_rolling_filter(exp['strengths'], window_size)
                
            selected_trials=np.nonzero(parameter_trials==param)[0]
            
            if 'alpha' in exp:
                alpha=exp['alpha']
            else:
                alpha=prot_ps['alpha']

            plt.close('all')
            for trial in selected_trials:
                fig,ax=plt.subplots(dpi=75)

                tmin=prot_ps['pause_start_times'][trial]
                thalf=prot_ps['flow_start_times'][trial]
                tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
                cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
                fliptimes_trial=tuple([(exp['fliptimes']>tmin)&(exp['fliptimes']<tmax)])
                ax.plot(exp['cameratimes'][cameratimes_trial],strengths_filt[cameratimes_trial])
                ax.plot(exp['fliptimes'][fliptimes_trial],(exp['v_flow']/alpha)[fliptimes_trial])
                ax.set_xlim(tmin,tmax)
                if log:
                    ax.set_yscale('log')
                if ymax is not None:
                    ax.set_ylim(ymin,ymax)
                plt.show()

def plot_trials_experiments_autocorrelation(experiments,parameter_trials_str,e0=0,p0=0,ymax=None):

    @interact(e=(0,len(experiments)-1,1))
    def plot1(e=e0):

        path,date,run=experiments[e]

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            proj_ps=save_group_as_dict(f,'proj_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        parameter_trials=prot_ps[parameter_trials_str]
        params=np.unique(parameter_trials)

        @interact(p=(0,len(params)-1,1))
        def plot2(p=p0):

            param=params[p]
            print(param)

            strengths_filt=get_strengths_filt(exp,cam_ps)
            selected_trials=np.nonzero(parameter_trials==param)[0]
            
            if 'alpha' in exp:
                alpha=exp['alpha']
            else:
                alpha=prot_ps['alpha']

            plt.close('all')
            for trial in selected_trials:
                fig,ax=plt.subplots(dpi=75)

                tmin=prot_ps['pause_start_times'][trial]
                thalf=prot_ps['flow_start_times'][trial]
                tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
                cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
                fliptimes_trial=tuple([(exp['fliptimes']>tmin)&(exp['fliptimes']<tmax)])
                ax.plot(exp['cameratimes'][cameratimes_trial],strengths_filt[cameratimes_trial])
                ax.plot(exp['fliptimes'][fliptimes_trial],(exp['v_flow']/alpha)[fliptimes_trial])
                ax.set_xlim(tmin,tmax)
                if ymax is not None:
                    ax.set_ylim(0,ymax)
                plt.show()
                
                
                
def plot_trials_experiments_orientation(experiments,parameter_trials_str,e0=0,p0=0):

    @interact(e=(0,len(experiments)-1,1))
    def plot1(e=e0):

        path,date,run=experiments[e]

        filenames=get_filenames(path,date,run)

        with h5py.File(filenames['visual_flow'],'r') as f:
            exp=save_group_as_dict(f,'exp')
            prot_ps=save_group_as_dict(f,'prot_ps')
            proj_ps=save_group_as_dict(f,'proj_ps')
            cam_ps=save_group_as_dict(f,'cam_ps')

        parameter_trials=prot_ps[parameter_trials_str]
        params=np.unique(parameter_trials)
        
        v_flow_angle_nan=np.copy(exp['v_flow_angle'])
        v_flow_angle_nan[exp['v_flow']==0]=np.nan


        @interact(p=(0,len(params)-1,1))
        def plot2(p=p0):

            param=params[p]
            print(param)

            selected_trials=np.nonzero(parameter_trials==param)[0]

            plt.close('all')
            for trial in selected_trials:
                fig,ax=plt.subplots(dpi=75)

                tmin=prot_ps['pause_start_times'][trial]
                thalf=prot_ps['flow_start_times'][trial]
                tmax=prot_ps['flow_start_times'][trial]+prot_ps['flow_duration_trials'][trial]
                cameratimes_trial=tuple([(exp['cameratimes']>tmin)&(exp['cameratimes']<tmax)])
                fliptimes_trial=tuple([(exp['fliptimes']>tmin)&(exp['fliptimes']<tmax)])                
                ax.plot(exp['fliptimes'][fliptimes_trial],exp['oris'][fliptimes_trial])
                ax.plot(exp['fliptimes'][fliptimes_trial],v_flow_angle_nan[fliptimes_trial]+exp['oris_pause'][fliptimes_trial])

                ax.set_xlim(tmin,tmax)
                plt.show()

                
def plot_trials_around_transitions(exp,prot_ps,times_around_transition,strength_trials,strength_trial_average,strength_trial_std,set_str='v_set',tau=None,xmax=None,log_scale=False):

    fig=plt.figure(figsize=(6,6),dpi=150)
    gs = fig.add_gridspec(len(prot_ps[set_str]),len(prot_ps[set_str]), hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for i in range(len(prot_ps[set_str])):
        for j in range(len(prot_ps[set_str])):
            ax=axs[i,j]
            if j!=len(prot_ps[set_str])-1-i:
                t=np.where((prot_ps['all_transitions']==(prot_ps[set_str][j],prot_ps[set_str][len(prot_ps[set_str])-1-i])).all(axis=-1))[0][0]

                for s in range(len(strength_trials[t])):
                    ax.plot(times_around_transition,strength_trials[t][s],alpha=0.25,c='tab:blue',lw=0.25)

                ax.plot(times_around_transition,strength_trial_average[t],label='fish')
                ax.fill_between(times_around_transition,strength_trial_average[t]+strength_trial_std[t],strength_trial_average[t]-strength_trial_std[t],alpha=0.25)
                if set_str=='v_set':
                    ax.plot((times_around_transition[0],0,0,times_around_transition[-1]),(1/prot_ps['alpha']*prot_ps['all_transitions'][t][0],)*2+(1/prot_ps['alpha']*prot_ps['all_transitions'][t][1],)*2,label='current',zorder=0)
                    ax.set_yticks(prot_ps['v_set']/prot_ps['alpha'])
                    ax.set_ylim(0.5*np.min(prot_ps['v_set']/prot_ps['alpha']),1.5*np.max(prot_ps['v_set']/prot_ps['alpha']))

                elif set_str=='alpha_set':
                    ax.plot((times_around_transition[0],0,0,times_around_transition[-1]),(prot_ps['v']/prot_ps['all_transitions'][t][0],)*2+(prot_ps['v']/prot_ps['all_transitions'][t][1],)*2,label='current',zorder=0)
                    ax.set_yticks(prot_ps['v']/prot_ps['alpha_set'])
                    ax.set_ylim(0.5*np.min(prot_ps['v']/prot_ps['alpha_set']),1.5*np.max(prot_ps['v']/prot_ps['alpha_set']))

                if tau is not None:
                    ax.axvline(tau,color='k',linestyle='--')
                    ax.axvline(2*tau,color='k',linestyle='--')
                #ax.set_xticks([-1,0,1])
                handles, labels = ax.get_legend_handles_labels()

            else:
                #axs[i,j].remove()
                ax.set_facecolor('lightgray')

            if i==0 and j==len(prot_ps[set_str])-1:
                ax.legend(handles,labels,loc='center',framealpha=1)

    if xmax is None:
        xmax=times_around_transition[-1]
    ax.set_xlim(-xmax,xmax)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.show()