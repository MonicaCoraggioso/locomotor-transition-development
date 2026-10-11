import numpy as np
from functions.utilities import *
from scipy.optimize import curve_fit
from functions.visual_flow_behavior_analysis import *
from scipy.optimize import brentq
from scipy.special import lambertw,spence,factorial


def integrate_nonlinear_delay_equation_external_current(v0,T,omegaext,alpha,tau=0.15,r=1.6,omegac=0.07,dt=1e-3):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        vnew=vs[-1]+vs[-1]*r*logstar(omegaext-alpha*vds[-1],omegac)*dt        
        vdnew=vs[max(0,len(vs)-l)]

        vs.append(vnew)
        vds.append(vdnew)

    vs=np.array(vs)    
    vds=np.array(vds)

    return tt,vs,vds,l

    
def interbout_duration(ome,phii=0.9,omth=0.05):
    return phii/np.log(ome/omth)*np.heaviside(ome-omth,0)
    
def simulate_trajectory_intermittent(v0,ome,phii,omc,sigmai,alpha,r,Tb,sigmaxi,omth,lmbdc=0.01,n_bouts=1000,vmax=np.inf,drift_correction=None):
    vs=[v0]
    phi=0
    Tis=np.random.lognormal(mean=np.log(interbout_duration(ome,phii,omth)),sigma=sigmai,size=n_bouts)
    for n in range(n_bouts):
        if drift_correction=='lin':
            vnew=next_speed_log(vs[-1],ome=ome,alpha=alpha,r=r,omc=omc,Tb=Tb,Ti=Tis[n])*np.exp(sigmaxi*np.sqrt(Tb
                +Tis[n])*np.random.normal())*np.exp(r*lmbdc*(phi*(Tb+Tis[n])+Tb**2/2*(ome-alpha*vs[-1])+
                                                             Tis[n]**2/2*ome+Tb*Tis[n]*(ome-alpha*vs[-1])))
            phi+=Tb*(ome-alpha*vs[-1])+Tis[n]*ome
        elif drift_correction=='log': 
            vnew=next_speed_log(vs[-1],ome=ome,alpha=alpha,r=r,omc=omc,Tb=Tb,Ti=Tis[n])*np.exp(sigmaxi*np.sqrt(Tb
                +Tis[n])*np.random.normal())*np.exp(r*lmbdc*(phi*(Tb+Tis[n])+Tb**2/2*logstar((ome-alpha*vs[-1])/omc)+
                                                             Tis[n]**2/2*logstar(ome/omc)+Tb*Tis[n]*logstar((ome-alpha*vs[-1])/omc)))
            phi+=Tb*logstar((ome-alpha*vs[-1])/omc)+Tis[n]*logstar(ome/omc)
        else:
            vnew=next_speed_log(vs[-1],ome=ome,alpha=alpha,r=r,omc=omc,Tb=Tb,Ti=Tis[n])*np.exp(sigmaxi*np.sqrt(Tb+Tis[n])*np.random.normal())
        vs.append(vnew)
        if vs[-1]>vmax:
            vs[-1]=vmax
    vs=vs[1:]
    return Tis,vs

def exp_decay(t,Tc,x0):
    return x0*np.exp(-t/Tc)
    
def omegaqss(t,omega0,Tc=1,omegac=0.05):
    return omega0*np.exp(-t/Tc)/(1+np.abs(omega0/omegac)*(1-np.exp(-t/Tc)))

def fixed_point_lin(ome=1,alpha=1,Tb=0.2,Ti=0.3):
    return ome/alpha*(1+Ti/Tb)

def next_speed_lin(v,ome=1,alpha=1,Tb=0.2,Ti=0.3,k=5):
    return v+k*(ome-alpha*v)*Tb+k*ome*Ti
    
def next_speed_log(v,ome=1,alpha=1,r=1.5,omc=0.1,Tb=0.2,Ti=0.3):
    return v*(1+np.abs(ome-alpha*v)/omc)**(np.sign(ome-alpha*v)*r*Tb)*(1+ome/omc)**(r*Ti)

def fixed_point_log(ome=1,alpha=1,r=1.5,omc=0.1,Tb=0.2,Ti=0.3):
    return omc/alpha*((1+ome/omc)**(Ti/Tb)-1+ome/omc)

def plot_cobweb(ax,vs,c='r',lwc=0.5,alphac=1,ax_start=True):
    for n in range(len(vs)-1):
        if n==0 and ax_start:
            ax.plot([vs[n],vs[n]],[0,vs[n+1]],c=c,lw=lwc,alpha=alphac)
        else:
            ax.plot([vs[n],vs[n]],[vs[n],vs[n+1]],c=c,lw=lwc,alpha=alphac)
        ax.plot([vs[n],vs[n+1]],[vs[n+1],vs[n+1]],c=c,lw=lwc,alpha=alphac)

def fixed_point_ratio(om,rho):
    return (1+1/om*((1+om)**rho-1))/(1+rho)

def slope_at_fixed_point(om,rho,rtb):
    return 1-rtb*(1+(om-1)/(om+1)**rho)

def rtb_at_unstable_fixed_point(om,rho):
    return 2/(1+(om-1)/(om+1)**rho)

def rtb_at_oscillating_fixed_point(om,rho):
    return 1/(1+(om-1)/(om+1)**rho)

def integrate_delay_equation_linear_nonconstant_current(s0,T,tau,vc,alpha,k,dt=1e-3,fraction_for_parameter_estimation=0.5,sigma=0,p0=(1,0,1,0),smax=np.inf,smin=-np.inf):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        if np.isscalar(vc):
            vcnow=vc
        else:
            vcnow=vc[i]
        wv=k*(vcnow-alpha*sds[-1])  
        
        if (ss[-1]>smax and wv>0) or (ss[-1]<smin and wv<0):
            wv=0

        snew=ss[-1]+wv*dt+sigma*np.sqrt(dt)*np.random.normal()
        
        sdnew=ss[max(0,len(ss)-l)]

        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)
    
    
    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    
    if np.isscalar(vc):
        vcf=vc
    else:
        vcf=vc(tt[-i_est:-1])

    zeros=np.nonzero((ss[-i_est:-1]-vcf/alpha)*(ss[-i_est+1:]-vcf/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))


    return tt,ss,sds,A,T,l







def delay_unit_solution(t,k=1,tau=1,N=30):
    x = np.zeros_like(t)
    for n in range(N):
        term_mask = t >= n * tau
        t_shifted = t[term_mask] - n * tau
        term = ((-k)**n / factorial(n)) * t_shifted**n
        x[term_mask] += term
    return x
    

def delay_rescaled_solution(t,k,tau,vi=150,vt=40,t0=0,N=30,set_to_zero=1):
    t=t-t0
    x = np.zeros_like(t)
    for n in range(int(N)):
        term_mask = t >= n * tau
        t_shifted = t[term_mask] - n * tau
        term = ((-k)**n / factorial(n)) * t_shifted**n
        x[term_mask] += term
    x_r=vt+(vi-vt)*x
    if bool(set_to_zero):
        x_r[np.cumsum(x_r<=0).astype(bool)] = 0
    x_r[t<=0]=0
    return x_r



def fit_speed_onsets_for_all_fish(age_id,speed_id,time,fish_stats,k0=10,tau0=0.1,vt0=30,taumin=0.05,taumax=0.5,vtmin=20,vtmax=60,maxtime=0.75,valueafter=-50,crop=False,add_after=False,add_point=False,varyvt=True):

    param_values={'k':[],'tau':[],'vi':[],'vt':[]}
    
    for fish_id in list(fish_stats[age_id]['vfish'].keys()):
    
        med=fish_stats[age_id][speed_id][fish_id]['median'] 
    
        params = Parameters()
        params.add('k',value=k0,min=0,vary=1)
        params.add('tau',value=tau0,min=taumin,max=taumax,vary=1)
        params.add('vi',value=np.mean(med[(time>0)&(time<0.1)]),min=0,vary=1)
        params.add('vt',value=vt0,min=vtmin,max=vtmax,vary=varyvt)
        params.add('t0',value=0,vary=0)
        params.add('N',value=30,vary=0)
        params.add('set_to_zero',value=1,vary=0)

        model = Model(delay_rescaled_solution)
        if crop:
            result = model.fit(med[time<maxtime], params, t=time[time<maxtime])   
        elif add_after:
            result = model.fit(np.concatenate([med,np.zeros(len(time))+valueafter]), params, 
                           t=np.concatenate([time,time[-1]+time-time[0]]))   
        elif add_point:
            result = model.fit(np.concatenate([med,[add_point]]),params, 
                           t=np.concatenate([time,[time[-1]]]))   

        else:
            result = model.fit(med, params, t=time)   
    
        for name, param in result.params.items():
            print(f"{name} = {param.value}")    

        param_values['k'].append(result.params['k'].value)
        param_values['tau'].append(result.params['tau'].value)
        param_values['vi'].append(result.params['vi'].value)
        param_values['vt'].append(result.params['vt'].value)
        
        result.plot_fit()
        plt.xlim(time[0],time[-1])
        plt.ylim(bottom=0)
        plt.show()

    return param_values


def effective_g(g,xeff):
    return g*np.log(1+xeff)/xeff

def effective_amplitude(g,geff):
    if g>geff:
        return np.real(-g/geff*lambertw(-geff/g*np.exp(-geff/g),-1)-1)
    else:
        return 0

def stability_transition_amplitude(g):
    return effective_amplitude(g,np.pi/2)

def oscillation_transition_amplitude(g):
    return effective_amplitude(g,1/np.e)

def implicit_definition_noise_induced_amplitude(x,g,sigma):
    return np.log(1+x)-sigma**2/(2*g*x)

def implicit_definition_noise_induced_amplitude(x,g,sigma):
    return effective_g(g,x)-sigma**2/(2*x**2)

def noise_induced_amplitude(g,sigma,xmin=1e-15,xmax=1e15):
    return brentq(implicit_definition_noise_induced_amplitude,xmin,xmax,args=(g,sigma))

def implicit_definition_sigma_transition(sigma,g):
    return noise_induced_amplitude(g,sigma)-stability_transition_amplitude(g)

def sigma_transition(g,sigmamin=1e-5,sigmamax=1e5):
    if g>np.pi/2:
        return brentq(implicit_definition_sigma_transition,sigmamin,sigmamax,args=(g,))
    else:
        return 0
    
def std_theory_scaling_factors(g,sigma,clc=0.9,cni=0.7):
    std_lc=stability_transition_amplitude(g)*clc
    std_ni=noise_induced_amplitude(g,sigma/cni)*cni
    std_theory=std_lc+std_ni
    return std_theory

def effective_g_scaling_factors(g,sigma,clc=0.9,cni=0.7):
    std_lc=stability_transition_amplitude(g)*clc
    std_ni=noise_induced_amplitude(g,sigma/cni)*cni
    std_sum=std_lc+std_ni
    cmean=(std_lc*clc+std_ni*cni)/(std_lc+std_ni)
    return effective_g(g,std_sum/cmean)

def effective_g_parameters(a,b,tau,vc,sigma):
    return effective_g_scaling_factors(b*tau*vc,sigma*vc*b/a*np.sqrt(tau))


def frequency_theory_scaling_factors(g,sigma,clc=0.9,cni=0.7):
    geff_theory=effective_g_scaling_factors(g,sigma,clc=clc,cni=cni)
    char_freqs_theory=np.imag(lambertw(-geff_theory))/(2*np.pi)    
    return char_freqs_theory

def implicit_definition_sigma_for_geff(sigma,g,geff):
    return effective_g_scaling_factors(g,sigma)-geff

def sigma_for_geff(g,geff,sigmamin=1e-5,sigmamax=1e5):
    if implicit_definition_sigma_for_geff(sigmamin,g,geff)*implicit_definition_sigma_for_geff(sigmamax,g,geff)>0:
        return 0
    else:
        return brentq(implicit_definition_sigma_for_geff,sigmamin,sigmamax,args=(g,geff))
    
def implicit_definition_sigma_for_same_std(sigma,g,clc=0.9,cni=0.7):
    return noise_induced_amplitude(g,sigma/cni)*cni-stability_transition_amplitude(g)*clc

def sigma_for_same_std(g,sigmamin=1e-5,sigmamax=1e5):
    if g>np.pi/2:
        return brentq(implicit_definition_sigma_for_same_std,sigmamin,sigmamax,args=(g,))
    else:
        return 0    
    
    

def integrate_double_low_pass_equation(s0,T,k,tau=1,D=0,st=0,dt=1e-3,fraction_for_parameter_estimation=0.2,multiplicative=False):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    fs=[s0]
    us=[s0]
    for i in range(len(tt)-1):
        snew=ss[-1]+dt*(-ss[-1]+fs[-1])/tau
        fnew=fs[-1]+dt*(-fs[-1]+us[-1])/tau
        unew=us[-1]+dt*(-k*(ss[-1]-st))
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        if multiplicative:
            unew+=eps*ss[-1]
        else:
            unew+=eps
        ss.append(snew)
        fs.append(fnew)
        us.append(unew)
    ss=np.array(ss)    
    fs=np.array(fs)    
    ss=np.array(ss)    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,fs,us,A,T

def integrate_low_pass_equation(s0,T,k,tau=1,D=0,st=0,dt=1e-3,fraction_for_parameter_estimation=0.2,multiplicative=False):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    us=[s0]
    for i in range(len(tt)-1):
        snew=ss[-1]+dt*(-ss[-1]+us[-1])/tau
        unew=us[-1]+dt*(-k*(ss[-1]-st))
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        if multiplicative:
            unew+=eps*ss[-1]
        else:
            unew+=eps
        ss.append(snew)
        us.append(unew)
    ss=np.array(ss)    
    ss=np.array(ss)    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,us,A,T

def integrate_delay_and_low_pass_equation(s0,v0,T,g,tau=1,dt=1e-3,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    vs=[v0]
    vds=[v0]
    
    for i in range(len(tt)-1):
        snew=ss[-1]-g*vds[-1]*dt
        vnew=vs[-1]+dt*(ss[-1]-vs[-1])/tau
        vdnew=vs[max(0,len(vs)-l)]
        
        ss.append(snew)
        vs.append(vnew)    
        vds.append(vdnew)
        
    ss=np.array(ss)    
    vs=np.array(vs)    
    vds=np.array(vds)    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,vs,A,T


def integrate_low_pass_and_nonlinearity_equation(v0,u0,T,alpha=1,taulp=1,r=1,D=0,dt=1e-3,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    us=[u0]
    vs=[v0]
    vds=[v0]
    
    for i in range(len(tt)-1):
        unew=us[-1]-(us[-1]-vds[-1])*dt/taulp+np.random.normal(scale=np.sqrt(2*D*dt))
        vnew=-alpha*weber_response_no_threshold(us[-1],k=1,r=r)
        vdnew=vs[max(0,len(vs)-l)]
        
        #unew=us[-1]-(us[-1]-vds[-1])*dt/taulp+np.random.normal(scale=np.sqrt(2*D*dt))
        #vnew=-alpha*us[-1]
        #if vnew>0:
        #    vnew*=2
        #else:
        #    vnew/=2
        #vdnew=weber_response_no_threshold(vs[max(0,len(vs)-l)],k=1,r=1)
        
        vs.append(vnew)    
        us.append(unew)    
        vds.append(vdnew)
        
    vs=np.array(vs)    
    us=np.array(us)    
    vds=np.array(vds)    

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,us,A,T




def integrate_low_pass_and_nonlinearity_derivative_equation(v0,u0,T,alpha=1,taulp=1,r=1,D=0,dt=1e-3,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    us=[u0]
    vs=[v0]
    vds=[v0]
    
    for i in range(len(tt)-1):
        unew=us[-1]-(us[-1]-vds[-1])*dt/taulp+np.random.normal(scale=np.sqrt(2*D*dt))
        vnew=vs[-1]-alpha*weber_response_no_threshold(us[-1],k=1,r=r)*dt
        vdnew=vs[max(0,len(vs)-l)]
        
        #unew=us[-1]-(us[-1]-vds[-1])*dt/taulp+np.random.normal(scale=np.sqrt(2*D*dt))
        #vnew=-alpha*us[-1]
        #if vnew>0:
        #    vnew*=2
        #else:
        #    vnew/=2
        #vdnew=weber_response_no_threshold(vs[max(0,len(vs)-l)],k=1,r=1)
        
        vs.append(vnew)    
        us.append(unew)    
        vds.append(vdnew)
        
    vs=np.array(vs)    
    us=np.array(us)    
    vds=np.array(vds)    

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,us,A,T


def integrate_low_pass_and_nonlinearity_asymmetric_convergence_equation(v0,u0,T,alpha=1,taulp=1,r=1,D=0,dt=1e-3,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    us=[u0]
    vs=[v0]
    vds=[v0]
    
    for i in range(len(tt)-1):
        unew=us[-1]-(us[-1]-weber_response_no_threshold(vds[-1],k=1,r=1))*dt/taulp+np.random.normal(scale=np.sqrt(2*D*dt))
                
        if -alpha*us[-1]>vs[-1]:
            vnew=vs[-1]+r*(-vds[-1]-alpha*us[-1])*dt
        else:
            vnew=vs[-1]+1/r*(-vds[-1]-alpha*us[-1])*dt
        vdnew=vs[max(0,len(vs)-l)]
        
        #unew=us[-1]-(us[-1]-vds[-1])*dt/taulp+np.random.normal(scale=np.sqrt(2*D*dt))
        #vnew=-alpha*us[-1]
        #if vnew>0:
        #    vnew*=2
        #else:
        #    vnew/=2
        #vdnew=weber_response_no_threshold(vs[max(0,len(vs)-l)],k=1,r=1)
        
        vs.append(vnew)    
        us.append(unew)    
        vds.append(vdnew)
        
    vs=np.array(vs)    
    us=np.array(us)    
    vds=np.array(vds)    

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,us,A,T


def integrate_speed_plus_low_pass_equation(v0,u0,T,alpha=1,taulp=1,r=1,D=0,dt=1e-3,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    us=[u0]
    vs=[v0]
    vds=[v0]
    
    for i in range(len(tt)-1):
        
        speed=weber_response_no_threshold(vds[-1],k=1,r=1)
        
        unew=us[-1]+(speed*dt-us[-1]*dt/taulp+np.random.normal(scale=np.sqrt(2*D*dt)))
        
        vnew=vs[-1]-alpha*asymmetry(speed+us[-1],r)*dt
                     
        vdnew=vs[max(0,len(vs)-l)]
        
        #unew=us[-1]-(us[-1]-vds[-1])*dt/taulp+np.random.normal(scale=np.sqrt(2*D*dt))
        #vnew=-alpha*us[-1]
        #if vnew>0:
        #    vnew*=2
        #else:
        #    vnew/=2
        #vdnew=weber_response_no_threshold(vs[max(0,len(vs)-l)],k=1,r=1)
        
        vs.append(vnew)    
        us.append(unew)    
        vds.append(vdnew)
        
    vs=np.array(vs)    
    us=np.array(us)    
    vds=np.array(vds)    

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,us,A,T


def integrate_double_low_pass_equation_non_zero(s0,vc,alpha,T,k,tau1=1,tau2=None,D=0,st=0,dt=1e-3,fraction_for_parameter_estimation=0.2,multiplicative=False):
    if tau2 is None:
        tau2=tau1
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    fs=[s0]
    us=[s0]
    for i in range(len(tt)-1):
        snew=ss[-1]+dt*(-ss[-1]+fs[-1])/tau2
        fnew=fs[-1]+dt*(-fs[-1]+us[-1])/tau1
        unew=k*(vc-alpha*ss[-1])
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        if multiplicative:
            unew+=eps*ss[-1]
        else:
            unew+=eps
        ss.append(snew)
        fs.append(fnew)
        us.append(unew)
    ss=np.array(ss)    
    fs=np.array(fs)    
    ss=np.array(ss)    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,fs,us,A,T

def integrate_double_integrator_equation_delay(s0,vc,alpha,T,k1,k2=None,tau1=1,tau2=None,rf=0.5,tau_d=0,D=0,st=0,dt=1e-3,fraction_for_parameter_estimation=0.2,multiplicative=False):
    if tau2 is None:
        tau2=tau1
    if k2 is None:
        k2=k1


    if tau_d==0:
        l=1
    else:
        l=int(round(tau_d/dt))

    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    fs=[s0]
    gs=[s0]
    us=[s0]
    for i in range(len(tt)-1):
        fnew=fs[-1]+dt*(-fs[-1]/tau1+k1*us[-1])
        gnew=gs[-1]+dt*(-gs[-1]/tau2+k2*us[-1])
        snew=rf*fnew+(1-rf)*gnew
        unew=vc-alpha*ss[max(0,len(ss)-l)]
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        if multiplicative:
            unew+=eps*ss[-1]
        else:
            unew+=eps
        ss.append(snew)
        fs.append(fnew)
        gs.append(gnew)
        us.append(unew)
    ss=np.array(ss)    
    fs=np.array(fs)    
    gs=np.array(gs)    
    ss=np.array(ss)    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,fs,gs,us,A,T

def integrate_integrator_acceleration_equation_delay(s0,vc,alpha,T,kv,ka,tau=1,tau_d=0,D=0,st=0,dt=1e-3,fraction_for_parameter_estimation=0.2,multiplicative=False):
    if tau_d==0:
        l=1
    else:
        l=int(round(tau_d/dt))

    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    acs=[s0]
    us=[vc-alpha*s0]
    for i in range(len(tt)-1):
        snew=ss[-1]+dt*(-ss[-1]/tau+kv*us[-1]+ka*acs[-1])
        acnew=(us[max(0,len(us)-1)]-us[max(0,len(us)-2)])/dt
        unew=vc-alpha*ss[max(0,len(ss)-l)]
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        if multiplicative:
            unew+=eps*ss[-1]
        else:
            unew+=eps
        ss.append(snew)
        acs.append(acnew)
        us.append(unew)
    ss=np.array(ss)    
    acs=np.array(acs)    
    ss=np.array(ss)    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,acs,us,A,T
    
def integrate_double_low_pass_equation_delay(s0,T,k,tau=1,tau_d=0,D=0,st=0,dt=1e-3,fraction_for_parameter_estimation=0.2,multiplicative=False):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    if tau_d==0:
        l=1
    else:
        l=int(round(tau_d/dt))
    ss=[s0]
    fs=[s0]
    us=[s0]
    for i in range(len(tt)-1):
        snew=ss[-1]+dt*(-ss[-1]+fs[-1])/tau
        fnew=fs[-1]+dt*(-fs[-1]+us[-1])/tau
        unew=us[-1]+dt*(-k*(ss[max(0,len(ss)-l)]-st))
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        if multiplicative:
            unew+=eps*ss[-1]
        else:
            unew+=eps
        ss.append(snew)
        fs.append(fnew)
        us.append(unew)
    ss=np.array(ss)    
    fs=np.array(fs)    
    ss=np.array(ss)    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,fs,us,A,T


def integrate_delay_equation_limited_range_noise(s0,T,k,alpha=1,tau=1,D=0,st=0,dt=1e-3,sm=0.5,sp=5,fraction_for_parameter_estimation=0.2,multiplicative=False):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        snew=ss[-1]+k*alpha*(st-sds[-1])*dt
        

        if snew<sm:
            snew=sm
        if snew>sp:
            snew=sp        
            
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        if multiplicative:
            snew+=eps*ss[-1]
        else:
            snew+=eps
        
        ss.append(snew)
        sds.append(ss[max(0,len(ss)-l)])
    ss=np.array(ss)    
    sds=np.array(sds)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero((ss[-i_est:-1]-st)*(ss[-i_est+1:]-st)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,sds,A,T,l



def integrate_delay_equation_sigmoid(v0,T,g,r=1,dt=1e-3,fraction_for_parameter_estimation=0.2,sigma=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        vs.append(vs[-1]+g*asymmetric_sigmoid(-vds[-1],ap=1,am=r,k=1)*dt+np.random.normal(scale=sigma*np.sqrt(dt)))
        vds.append(vs[max(0,len(vs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,A,T,l

def integrate_delay_equation_weber_log(s0,T,k,tau,vmin,vmax,vc,alpha,dt=1e-3,fraction_for_parameter_estimation=0.5,D=0,p0=(1,1,1,0)):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        ss.append(ss[-1]+ss[-1]*weber_response(vc-alpha*sds[-1],x0=vmin,xmax=vmax,k=k)*dt+ss[-1]*np.random.normal(scale=np.sqrt(2*D*dt)))
        sds.append(ss[max(0,len(ss)-l)])
    ss=np.array(ss)    
    sds=np.array(sds)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero((ss[-i_est:-1]-vc/alpha)*(ss[-i_est+1:]-vc/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))
        
        
    params = Parameters()
    params.add('a', value=p0[0],min=0)
    params.add('lmbd', value=p0[1])
    params.add('omega', value=p0[2],min=0)
    params.add('phi', value=p0[3], min=0, max=2*np.pi)

    dmodel = Model(damped_cosine)
    result = dmodel.fit(ss[2*l:], params, t=tt[2*l:])

    return tt,ss,sds,A,T,result.params['lmbd'].value,result.params['omega'].value,l


def integrate_delay_equation_weber_asymmetric(s0,T,x0,k1,k2,b,tau,vc,alpha,dt=1e-3,fraction_for_parameter_estimation=0.5,D=0,p0=(1,1,1,0)):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        ss.append(ss[-1]+ss[-1]*weber_response_asymmetric(vc-alpha*sds[-1],x0,k1,k2,b)*dt+ss[-1]*np.random.normal(scale=np.sqrt(2*D*dt)))
        sds.append(ss[max(0,len(ss)-l)])
    ss=np.array(ss)    
    sds=np.array(sds)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero((ss[-i_est:-1]-vc/alpha)*(ss[-i_est+1:]-vc/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))
        
        
    params = Parameters()
    params.add('a', value=p0[0],min=0)
    params.add('lmbd', value=p0[1])
    params.add('omega', value=p0[2],min=0)
    params.add('phi', value=p0[3], min=0, max=2*np.pi)

    dmodel = Model(damped_cosine)
    result = dmodel.fit(ss[2*l:], params, t=tt[2*l:])

    return tt,ss,sds,A,T,result.params['lmbd'].value,result.params['omega'].value,l


def integrate_delay_equation_weber_low_pass_term(s0,T,x0,b,tau,vc,alpha,Ap,Am,taulp,klp,dt=1e-3,fraction_for_parameter_estimation=0.5,D=0,p0=(1,1,1,0),smax=10,smin=0.1,wvlp0=0,xmax=1,gamma=1):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    wvlps=[wvlp0]
    wvs=[0]
    for i in range(len(tt)-1):
        #wv=weber_response_asymmetric(vc-alpha*sds[-1],x0,1,1,0)
        if np.isscalar(vc):
            vcnow=vc
        else:
            vcnow=vc[i]
        wv=weber_response(vcnow-alpha*sds[-1],x0,xmax,1)
        wvlpnew=wvlps[-1]+dt*(-wvlps[-1]+wv)**gamma/taulp
        A=wv+klp*wvlpnew+b#+np.random.normal(scale=np.sqrt(D))
        #if np.abs(A)<1e-5:
        #    A=b
        if A>0:
            A=Ap*A
        elif A<0:
            A=Am*A
            
        
        snew=ss[-1]+ss[-1]*A*dt+b*ss[-1]*np.log(ss[-1])*dt+ss[-1]*np.random.normal(scale=np.sqrt(2*D*dt))
        
        if snew>smax:
            snew=smax
        if snew<smin:
            snew=smin
        
        sdnew=ss[max(0,len(ss)-l)]

        wvs.append(wv)
        wvlps.append(wvlpnew)
        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)
    wvs=np.array(wvs)
    wvlps=np.array(wvlps)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero((ss[-i_est:-1]-vcnow/alpha)*(ss[-i_est+1:]-vcnow/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))
        
        
    params = Parameters()
    params.add('a', value=p0[0],min=0)
    params.add('lmbd', value=p0[1])
    params.add('omega', value=p0[2],min=0)
    params.add('phi', value=p0[3], min=0, max=2*np.pi)

    dmodel = Model(damped_cosine)
    result = dmodel.fit(ss[2*l:], params, t=tt[2*l:])

    return tt,ss,sds,wvs,wvlps,A,T,result.params['lmbd'].value,result.params['omega'].value,l


def perform_robust_regression(ss,dt,vc,alpha,tau,sigma_gaussian=0.06,thr_vc=0.5,n_resamples=100,n_points=1000):
    ss_filt=gaussian_filter(ss,sigma_gaussian/dt)
    ss_gradient=np.gradient(ss_filt,dt)
    logss_gradient=ss_gradient/ss_filt
    v_shifted=np.roll(vc-alpha*ss_filt,round(tau/dt))
    

    slopes_tsbs=np.zeros(n_resamples)
    mask=(np.abs(v_shifted)<thr_vc*vc)
    x=v_shifted[mask]
    y=ss_gradient[mask]
    for n in range(n_resamples):
        random_indices=np.random.randint(len(y),size=n_points)
        slopes_tsbs[n]=theilslopes(y[random_indices],x[random_indices])[0]

    slopes_tsbs_mean=np.mean(slopes_tsbs,axis=-1)
    slopes_tsbs_std=np.std(slopes_tsbs,axis=-1)
    
    return slopes_tsbs_mean,slopes_tsbs_std

    

def perform_robust_regression_simulation(ss,dt,vc,alpha,tau,sigma_strength=0.06,sigma_flow=0.02,n_points=1000,min_quantile=0.1,max_quantile=0.9):
    ss_filt=gaussian_filter(ss,sigma_strength/dt)
    ss_gradient=np.gradient(ss_filt,dt)
    #logss_gradient=ss_gradient/ss_filt
    v_shifted=gaussian_filter(np.roll(vc-alpha*ss,round(tau/dt)),sigma_flow/dt)

    x=v_shifted
    y=ss_gradient
    mask=(x>np.quantile(x,min_quantile))&(x<np.quantile(x,max_quantile))
    x=x[mask]
    y=y[mask]
    random_indices=np.random.randint(len(y),size=n_points)    
    results=theilslopes(y[random_indices],x[random_indices])
    slope=results[0]
    slopem=results[2]
    slopep=results[3]

    return slope,slopem,slopep


def perform_robust_regression_simulation_bootstrap(ss,dt,vc,alpha,tau,sigma_strength=0.06,sigma_flow=0.02,n_resamples=100,n_points=1000,min_quantile=0.1,max_quantile=0.9):
    ss_filt=gaussian_filter(ss,sigma_strength/dt)
    ss_gradient=np.gradient(ss_filt,dt)
    #logss_gradient=ss_gradient/ss_filt
    v_shifted=gaussian_filter(np.roll(vc-alpha*ss,round(tau/dt)),sigma_flow/dt)

    x=v_shifted
    y=ss_gradient
    mask=(x>np.quantile(x,min_quantile))&(x<np.quantile(x,max_quantile))
    x=x[mask]
    y=y[mask]
    slopes_tsbs=np.zeros(n_resamples)
    for n in range(n_resamples):
        random_indices=np.random.randint(len(y),size=n_points)
        slopes_tsbs[n]=theilslopes(y[random_indices],x[random_indices])[0]

    slopes_tsbs_mean=np.mean(slopes_tsbs,axis=-1)
    slopes_tsbs_std=np.std(slopes_tsbs,axis=-1)


    return slopes_tsbs_mean,slopes_tsbs_std


def perform_simple_regression_simulation(ss,dt,vc,alpha,tau,sigma_strength=0.06,sigma_flow=0.02,min_quantile=0.1,max_quantile=0.9):
    ss_filt=gaussian_filter(ss,sigma_strength/dt)
    ss_gradient=np.gradient(ss_filt,dt)
    #logss_gradient=ss_gradient/ss_filt
    v_shifted=gaussian_filter(np.roll(vc-alpha*ss,round(tau/dt)),sigma_flow/dt)

    x=v_shifted
    y=ss_gradient
    mask=(x>np.quantile(x,min_quantile))&(x<np.quantile(x,max_quantile))
    x=x[mask]
    y=y[mask]
    slope=factor_that_minimizes_mse(x,y)

    return slope
    

def calculate_spectrogram(ss,dt,segment_duration=10,segment_overlap=0.5,sigma_strength=0.06,mode='default'):

    strengths_filt=gaussian_filter(ss,sigma_strength/dt)
    strengths_change_filt=np.gradient(strengths_filt,dt)
    log_strengths_change_filt=strengths_change_filt/strengths_filt
    if mode=='log_change':
        signal=log_strengths_change_filt
    elif mode=='change':
        signal=strengths_change_filt
    else:
        signal=strengths_filt
        
    f, t, spectrum = spectrogram(signal, 1/dt, nperseg=int(segment_duration/dt),noverlap=int(segment_overlap*segment_duration/dt))
    
    return f,t,spectrum   
    
    

def integrate_delay_equation_weber_nonconstant_current(s0,T,tau,vc,alpha,x0,k1,k2,b,xmax,dt=1e-3,fraction_for_parameter_estimation=0.5,D=0,p0=(1,1,1,0),smax=10,smin=0.1):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        if np.isscalar(vc):
            vcnow=vc
        else:
            vcnow=vc[i]
        wv=weber_response_asymmetric(vcnow-alpha*sds[-1],x0,k1,k2,b,xmax)    
        snew=ss[-1]+ss[-1]*wv*dt+ss[-1]*np.random.normal(scale=np.sqrt(2*D*dt))
        
        if snew>smax:
            snew=smax
        if snew<smin:
            snew=smin
        
        sdnew=ss[max(0,len(ss)-l)]

        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)

    return tt,ss,sds,l


def integrate_delay_equation_weber_cont_sym_nonconstant_current(s0,T,tau,vc,alpha,b,a,dt=1e-3,fraction_for_parameter_estimation=0.5,D=0,p0=(1,0,1,0),smax=10,smin=0.1):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        if np.isscalar(vc):
            vcnow=vc
        else:
            vcnow=vc[i]
        wv=weber_response_cont_sym(vcnow-alpha*sds[-1],b,a)  
        
        if (ss[-1]>smax and wv>0) or (ss[-1]<smin and wv<0):
            wv=0

        snew=ss[-1]+ss[-1]*wv*dt+ss[-1]*np.random.normal(scale=np.sqrt(2*D*dt))
        

        
        sdnew=ss[max(0,len(ss)-l)]

        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)
    
    
    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    
    if np.isscalar(vc):
        vcf=vc
    else:
        vcf=vc(tt[-i_est:-1])

    zeros=np.nonzero((ss[-i_est:-1]-vcf/alpha)*(ss[-i_est+1:]-vcf/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))


    return tt,ss,sds,A,T,l


def integrate_delay_equation_weber_cont_sym_nonconstant_current_ito_strat(s0,T,tau,vc,alpha,b,a,dt=1e-3,ito=False,fraction_for_parameter_estimation=0.5,sigma=0,p0=(1,0,1,0),smax=np.inf,smin=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        if np.isscalar(vc):
            vcnow=vc
        else:
            vcnow=vc[i]
        wv=weber_response_cont_sym(vcnow-alpha*sds[-1],b,a)  
        
        if (ss[-1]>smax and wv>0) or (ss[-1]<smin and wv<0):
            wv=0

        if ito:
            snew=ss[-1]+ss[-1]*wv*dt+ss[-1]*sigma*np.sqrt(dt)*np.random.normal()
        else:
            snew=ss[-1]*np.exp(wv*dt+sigma*np.sqrt(dt)*np.random.normal())

        
        sdnew=ss[max(0,len(ss)-l)]

        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)
    
    
    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    
    if np.isscalar(vc):
        vcf=vc
    else:
        vcf=vc(tt[-i_est:-1])

    zeros=np.nonzero((ss[-i_est:-1]-vcf/alpha)*(ss[-i_est+1:]-vcf/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))


    return tt,ss,sds,A,T,l


def integrate_delay_equation_weber_cont_sym_nonconstant_current_ito_strat_kernel(s0,T,tau,vc,alpha,b,a,std_kernel=0.02,dt=1e-3,ito=False,fraction_for_parameter_estimation=0.5,sigma=0,p0=(1,0,1,0),smax=np.inf,smin=0):

    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    t_kernel=dt*np.arange(1,round(2*tau/dt))
    kernel=1/np.sqrt(2*np.pi*std_kernel**2)*np.exp(-(t_kernel-tau)**2/(2*std_kernel**2))

    for i in range(len(tt)-1):
        if np.isscalar(vc):
            vs_tmp=vc-alpha*np.array(ss[max(0,len(ss)-len(kernel)):max(0,len(ss))])
        else:
            vs_tmp=vc[max(0,len(ss)-len(kernel)):max(0,len(ss))]-alpha*np.array(ss[max(0,len(ss)-len(kernel)):max(0,len(ss))])
        v_conv=np.sum(vs_tmp*kernel[-len(vs_tmp):]*dt)
        wv=weber_response_cont_sym(v_conv,b,a)  
        
        if (ss[-1]>smax and wv>0) or (ss[-1]<smin and wv<0):
            wv=0

        if ito:
            snew=ss[-1]+ss[-1]*wv*dt+ss[-1]*sigma*np.sqrt(dt)*np.random.normal()
        else:
            snew=ss[-1]*np.exp(wv*dt+sigma*np.sqrt(dt)*np.random.normal())

        
        ss.append(snew)
    ss=np.array(ss)    
    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    
    if np.isscalar(vc):
        vcf=vc
    else:
        vcf=vc(tt[-i_est:-1])

    zeros=np.nonzero((ss[-i_est:-1]-vcf/alpha)*(ss[-i_est+1:]-vcf/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))


    return tt,ss,A,T


def integrate_delay_equation_weber_cont_sym_nonconstant_current_ito_strat_kernel_barrier(s0,T,tau,vc,alpha,b,a,tau_barrier=0.1,std_kernel=0.02,dt=1e-3,ito=False,fraction_for_parameter_estimation=0.5,sigma=0,p0=(1,0,1,0),smax=np.inf,smin=0):

    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    t_kernel=dt*np.arange(1,round(2*tau/dt))
    kernel=1/np.sqrt(2*np.pi*std_kernel**2)*np.exp(-(t_kernel-tau)**2/(2*std_kernel**2))

    for i in range(len(tt)-1):
        if np.isscalar(vc):
            vs_tmp=vc-alpha*np.array(ss[max(0,len(ss)-len(kernel)):max(0,len(ss))])
        else:
            vs_tmp=vc[max(0,len(ss)-len(kernel)):max(0,len(ss))]-alpha*np.array(ss[max(0,len(ss)-len(kernel)):max(0,len(ss))])
        v_conv=np.sum(vs_tmp*kernel[-len(vs_tmp):]*dt)
        wv=weber_response_cont_sym(v_conv,b,a)  
        
        if ito:
            snew=ss[-1]+ss[-1]*wv*dt+ss[-1]*sigma*np.sqrt(dt)*np.random.normal()
        else:
            snew=ss[-1]*np.exp(wv*dt+sigma*np.sqrt(dt)*np.random.normal()+np.heaviside(ss[-1]-smax,0)*(smax-ss[-1])/ss[-1]/tau_barrier*dt+np.heaviside(smin-ss[-1],0)*(smin-ss[-1])/ss[-1]/tau_barrier*dt)

        #if ss[-1]>smax:
        #    ss[-1]=smax
        #if ss[-1]<smin:
        #    ss[-1]=smin
                
        ss.append(snew)
    ss=np.array(ss)    
    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    
    if np.isscalar(vc):
        vcf=vc
    else:
        vcf=vc(tt[-i_est:-1])

    zeros=np.nonzero((ss[-i_est:-1]-vcf/alpha)*(ss[-i_est+1:]-vcf/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))


    return tt,ss,A,T


def integrate_delay_equation_custom_kernel_additional_delay(T,alpha,kernel_times,kernel_points,dt=1e-3,fraction_for_parameter_estimation=0.2,tau=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    l=round(tau/dt)
    kernel=np.interp(dt*np.arange(1,round(kernel_times[-1]/dt)),kernel_times,kernel_points)[::-1]
    ss=list(np.zeros(len(kernel)+l))
    vds=list(np.zeros(len(kernel)+l))
    vcs=list(np.zeros(len(kernel)+l))
    vcs[-1]=1
    
    for i in range(len(tt)-1):
        snew=ss[-1]+vds[-1]*dt
        vcnew=0
        #vs_tmp=vcs[max(0,len(vcs)-len(kernel)):]-alpha*np.array(ss[max(0,len(ss)-len(kernel)):])
        vs_tmp=vcs[max(0,len(vcs)-len(kernel)):]-alpha*np.array(ss[max(0,len(ss)-len(kernel)-l):max(0,len(ss)-l)])
        vdnew=np.sum(vs_tmp*kernel[-len(vs_tmp):]*dt)
        ss.append(snew)
        vcs.append(vcnew)
        vds.append(vdnew)
    ss=np.array(ss)    
    vds=np.array(vds)
    vcs=np.array(vcs)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,vds,vcs,A,T








def integrate_delay_equation_weber_cont_sym_nonconstant_current_bouts(s0,T,tau,vc,alpha,b=50,a=2,ibi=1,sstart=1,dt=1e-2,sigma=0,smax=5,smin=0.5):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_tau=int(round(tau/dt))
    i_ibi=int(round(ibi/dt))
    if np.isscalar(s0):    
        ss=[s0]*i_tau
    else:
        ss=s0
    vs=[np.nan]*i_tau
    
    bouts=False
    for i in np.arange(i_tau,len(tt)):
        
        if np.isscalar(vc):
            vcnow=vc
        else:
            vcnow=vc[i]

        v=vcnow-alpha*ss[len(ss)-i_tau]
        wv=weber_response_cont_sym(v,b,a)  

        if (ss[-1]>smax and wv>0) or (ss[-1]<smin and wv<0):
            wv=0

        if np.all(np.array(ss[-i_ibi:])==0):
            snew=sstart
        elif ss[-1]<smin:
            snew=0
            bouts=True
        else:
            snew=ss[-1]*np.exp(wv*dt+sigma*np.sqrt(dt)*np.random.normal())


        ss.append(snew)
        vs.append(v)
        
    ss=np.array(ss)    
    vs=np.array(vs)


    return tt,ss,vs,bouts




def integrate_delay_equation_weber_cont_sym_check_initial_transient(vc,alpha,tau=0.15,b=50,a=2,sstart=1,T=5,dt=1e-2,sigma=0,smax=np.inf,smin=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_tau=int(round(tau/dt))
    ss=[0]*(i_tau-1)+[sstart]
    vs=[vc]*(i_tau)
    for i in range(len(tt)-i_tau):
        
        v=vc-alpha*ss[len(ss)-i_tau]
        wv=weber_response_cont_sym(v,b,a)  
        if (ss[-1]>smax and wv>0) or (ss[-1]<smin and wv<0):
            wv=0            
        else:
            snew=ss[-1]*np.exp(wv*dt+sigma*np.sqrt(dt)*np.random.normal())
        ss.append(snew)
        vs.append(v)

    ss=np.array(ss)    
    vs=np.array(vs)

    return tt,ss,vs





def integrate_delay_equation_spontaneous_swimming(ui,ustar,k,tau,T=5,dt=1e-2,sigma=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_tau=int(round(tau/dt))
    us=[ui]*(i_tau)
    for i in range(len(tt)-i_tau):
            
        unew=us[-1]*np.exp(-k*np.log(us[len(us)-i_tau]/ustar)*dt+sigma*np.sqrt(dt)*np.random.normal())
        us.append(unew)

    us=np.array(us)    

    return tt,us






def integrate_delay_equation_weber_cont_sym_nondim(s0,T,g,dt=1e-3,fraction_for_parameter_estimation=0.5,sigma=0,tau=1,estimate_parameters=True):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    xis=np.random.normal(size=len(tt)-1)
    for i in range(len(tt)-1):
        wv=weber_response_cont_sym(sds[-1],1,1)    
        snew=ss[-1]-g*wv*dt+sigma*np.sqrt(dt)*xis[i]
        sdnew=ss[max(0,len(ss)-l)]

        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)
    
    if estimate_parameters:
        
        A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
        #zeros=np.nonzero((ss[-i_est:-1])*(ss[-i_est+1:])<0)[0]
        zeros=np.nonzero((ss[:-1-i_est])*(ss[+1:-i_est])<0)[0]
        if len(zeros)==0:
            T=np.nan
        else:
            T=np.mean(dt*(zeros[2:]-zeros[:-2]))
    
        return tt,ss,sds,A,T,l

    else:
        return tt,ss,sds,l


def integrate_delay_equation_linear_nondim(s0,T,g,dt=1e-3,fraction_for_parameter_estimation=0.5,sigma=0,tau=1):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        snew=ss[-1]-g*sds[-1]*dt+sigma*np.sqrt(dt)*np.random.normal()
        sdnew=ss[max(0,len(ss)-l)]

        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)
    
    
    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    

    #zeros=np.nonzero((ss[-i_est:-1])*(ss[-i_est+1:])<0)[0]
    zeros=np.nonzero((ss[:-1-i_est])*(ss[+1:-i_est])<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))


    return tt,ss,sds,A,T,l



def integrate_delay_equation_sigmoid_nondim(s0,T,g,dt=1e-3,fraction_for_parameter_estimation=0.5,):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        wv=np.tanh(sds[-1])    
        snew=ss[-1]-g*wv*dt
        sdnew=ss[max(0,len(ss)-l)]

        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)
    
    
    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    

    zeros=np.nonzero((ss[-i_est:-1])*(ss[-i_est+1:])<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))


    return tt,ss,sds,A,T,l

def integrate_delay_equation_two_slopes(s0,T,g1,g2,sthr,dt=1e-3,fraction_for_parameter_estimation=0.5,D=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    for i in range(len(tt)-1):
        if np.abs(sds[-1])<sthr:
            wv=g1*sds[-1]
        else:
            #wv=g2*sds[-1]
            #wv=(g1-g2)*sthr*np.sign(sds[-1])+g2*sds[-1]
            wv=np.sign(sds[-1])*np.max([g2*np.abs(sds[-1]),g1*sthr])
        snew=ss[-1]-wv*dt+np.random.normal(scale=np.sqrt(2*D*dt))
        sdnew=ss[max(0,len(ss)-l)]

        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)
    
    
    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    

    zeros=np.nonzero((ss[-i_est:-1])*(ss[-i_est+1:])<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))


    return tt,ss,sds,A,T,l


def integrate_delay_equation_step_asymmetry_low_pass_term(s0,T,tau,vc,alpha,Ap,Am,taulp,klp,dt=1e-3,fraction_for_parameter_estimation=0.5,D=0,p0=(1,1,1,0),smax=10,smin=0.1,vslp0=0,gamma=1):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    svlps=[vslp0]
    vs=[0]
    for i in range(len(tt)-1):
        v=vc-alpha*sds[-1]
        sv=np.sign(v)
        svlpnew=svlps[-1]+dt*(-svlps[-1]+sv)/taulp
        A=sv+klp*svlpnew#+np.random.normal(scale=np.sqrt(D))
        if A>0:
            A=Ap*A
        elif A<0:
            A=Am*A
            
        
        snew=ss[-1]+ss[-1]*A*dt+ss[-1]*np.random.normal(scale=np.sqrt(2*D*dt))
        
        if snew>smax:
            snew=smax
        if snew<smin:
            snew=smin
        
        sdnew=ss[max(0,len(ss)-l)]

        vs.append(v)
        svlps.append(svlpnew)
        ss.append(snew)
        sds.append(sdnew)
    ss=np.array(ss)    
    sds=np.array(sds)
    vs=np.array(vs)
    svlps=np.array(svlps)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero((ss[-i_est:-1]-vc/alpha)*(ss[-i_est+1:]-vc/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))
        
        
    params = Parameters()
    params.add('a', value=p0[0],min=0)
    params.add('lmbd', value=p0[1])
    params.add('omega', value=p0[2],min=0)
    params.add('phi', value=p0[3], min=0, max=2*np.pi)

    dmodel = Model(damped_cosine)
    result = dmodel.fit(ss[2*l:], params, t=tt[2*l:])

    return tt,ss,sds,vs,svlps,A,T,result.params['lmbd'].value,result.params['omega'].value,l




def integrate_delay_equation_weber_error_term(s0,T,x0,b,tau,vc,alpha,Ap,Am,taulp,klp,dt=1e-3,fraction_for_parameter_estimation=0.5,D=0,p0=(1,1,1,0),smax=10,smin=0.1,wvlp0=0,xmax=1,gamma=1):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sds=[s0]
    wvlps=[wvlp0]
    wvs=[0]
    errors=[0]
    for i in range(len(tt)-1):
        #wv=weber_response_asymmetric(vc-alpha*sds[-1],x0,1,1,0)
        wv=weber_response(vc-alpha*sds[-1],x0,xmax,1)
        wvlpnew=wvlps[-1]+dt*(-wvlps[-1]+wv**gamma)/taulp
        errornew=errors[-1]+dt*wv
        A=wv+klp*errornew#+np.random.normal(scale=np.sqrt(D))
        #if np.abs(A)<1e-5:
        #    A=b
        if A>0:
            A=Ap*A
        elif A<0:
            A=Am*A
            
        
        snew=ss[-1]+ss[-1]*A*dt+b*ss[-1]*np.log(ss[-1])*dt+ss[-1]*np.random.normal(scale=np.sqrt(2*D*dt))
        
        if snew>smax:
            snew=smax
        if snew<smin:
            snew=smin
        
        sdnew=ss[max(0,len(ss)-l)]

        wvs.append(wv)
        wvlps.append(wvlpnew)
        ss.append(snew)
        sds.append(sdnew)
        errors.append(errornew)
    ss=np.array(ss)    
    sds=np.array(sds)
    wvs=np.array(wvs)
    wvlps=np.array(wvlps)
    errors=np.array(errors)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero((ss[-i_est:-1]-vc/alpha)*(ss[-i_est+1:]-vc/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))
        
        
    params = Parameters()
    params.add('a', value=p0[0],min=0)
    params.add('lmbd', value=p0[1])
    params.add('omega', value=p0[2],min=0)
    params.add('phi', value=p0[3], min=0, max=2*np.pi)

    dmodel = Model(damped_cosine)
    result = dmodel.fit(ss[2*l:], params, t=tt[2*l:])

    return tt,ss,sds,wvs,wvlps,errors,A,T,result.params['lmbd'].value,result.params['omega'].value,l



def integrate_delay_equation_weber(v0,T,g,dt=1e-3,fraction_for_parameter_estimation=0.5,D=0,p0=(1,1,1,0)):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        vs.append(vs[-1]-g*weber_response(vds[-1],x0=1,xmax=np.inf,k=1)*dt+np.random.normal(scale=np.sqrt(2*D*dt)))
        vds.append(vs[max(0,len(vs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))
        
        
    params = Parameters()
    params.add('a', value=p0[0],min=0)
    params.add('lmbd', value=p0[1])
    params.add('omega', value=p0[2],min=0)
    params.add('phi', value=p0[3], min=0, max=2*np.pi)

    dmodel = Model(damped_cosine)
    result = dmodel.fit(vs[2*l:], params, t=tt[2*l:])

    return tt,vs,vds,A,T,result.params['lmbd'].value,result.params['omega'].value,l



def integrate_delay_equation_weber_no_threshold(v0,T,g,dt=1e-3,r=1,fraction_for_parameter_estimation=0.5,D=0,p0=(1,1,1,0)):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        vs.append(vs[-1]-g*weber_response_no_threshold(vds[-1],k=1,r=r)*dt+np.random.normal(scale=np.sqrt(2*D*dt)))
        vds.append(vs[max(0,len(vs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))
        
        
    params = Parameters()
    params.add('a', value=p0[0],min=0)
    params.add('lmbd', value=p0[1])
    params.add('omega', value=p0[2],min=0)
    params.add('phi', value=p0[3], min=0, max=2*np.pi)

    dmodel = Model(damped_cosine)
    result = dmodel.fit(vs[2*l:], params, t=tt[2*l:])

    return tt,vs,vds,A,T,result.params['lmbd'].value,result.params['omega'].value,l



def integrate_delay_equation_sigmoid_log(v0,T,q=0.3,vc=1,dt=1e-3,vm=None,vp=None,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        
        newv=vs[-1]+q*vs[-1]*asymmetric_sigmoid(vc-vds[-1],ap=1,am=1,k=1)*dt
        if vm is not None and newv<vm:
            newv=vm
        if vp is not None and newv>vp:
            newv=vp           
        vs.append(newv)
        vds.append(vs[max(0,len(vs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero((vs[-i_est:-1]-vc)*(vs[-i_est+1:]-vc)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,A,T,l




from lmfit import Model, Parameters

    
def damped_cosine(t, a, lmbd, omega, phi):
    return a*np.cos(omega*t+phi)*np.exp(lmbd*t)

def integrate_linear_delay_equation(v0,T,g,dt=1e-3,p0=(1,1,1,0),fraction_fit=0.5,sigma=0,fit=True):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    iif=int(fraction_fit*len(tt))
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        vs.append(vs[-1]-g*vds[-1]*dt+sigma*np.sqrt(dt)*np.random.normal())
        vds.append(vs[max(0,len(vs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    if fit:
        
        params = Parameters()
        params.add('a', value=p0[0],min=0)
        params.add('lmbd', value=p0[1])
        params.add('omega', value=p0[2],min=0)
        params.add('phi', value=p0[3], min=0, max=2*np.pi)
    
        dmodel = Model(damped_cosine)
        #result = dmodel.fit(vs[2*l:], params, t=tt[2*l:])
        result = dmodel.fit(vs[iif:], params, t=tt[iif:])
    
        return tt,vs,vds,result.params['lmbd'].value,result.params['omega'].value,l

    else:

        return tt,vs,vds

def integrate_ornstein_uhlenbeck_equation(v0,T,k,sigma=0,dt=1e-3):
    tt=np.linspace(0,T,int(T/dt)+1)
    vs=[v0]
    for i in range(len(tt)-1):
        vs.append(vs[-1]-k*vs[-1]*dt+sigma*np.sqrt(dt)*np.random.normal())
    vs=np.array(vs)    
    
    return tt,vs



def integrate_delay_equation_sigmoid_sinusoidal_forcing(v0,T,g,r=1,ampl=1,period=1,phi=0,D=0,
                                                        dt=1e-3,fraction_for_parameter_estimation=0.5):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    vcs=[ampl*np.sin(phi)]
    for i in range(len(tt)-1):
        vcs.append(ampl*np.sin(2*np.pi*tt[i]/period+phi))
        if D>0:
            eps=np.random.normal(scale=np.sqrt(2*D*dt))
        else:
            eps=0
        vs.append(vs[-1]+g*asymmetric_sigmoid(-vds[-1]+eps,ap=1,am=r,k=1)*dt)
        vds.append(vs[max(0,len(vs)-l)]-vcs[max(0,len(vcs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    crosscorr,dts=calculate_crosscorr(vs[-i_est:],vcs[-i_est:],dt)
    dt_shift=dts[np.argmax(crosscorr)]
    phaseshift=dt_shift/period*2*np.pi
    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,vcs,A,T,phaseshift,l


def integrate_delay_equation_limited_range(v0,T,g,tau=1,dt=1e-3,vm=-1,vp=1,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        newv=vs[-1]-g*vds[-1]*dt
        if newv<vm:
            newv=vm
        if newv>vp:
            newv=vp           
        vs.append(newv)
        vds.append(vs[max(0,len(vs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,A,T,l


def integrate_delay_equation_limited_range_log(v0,T,g,tau=1,dt=1e-3,vc=1,vm=0.1,vp=10,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        newv=vs[-1]+g*vs[-1]*(vc-vds[-1])*dt
        if newv<vm:
            newv=vm
        if newv>vp:
            newv=vp           
        vs.append(newv)
        vds.append(vs[max(0,len(vs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero((vs[-i_est:-1]-vc)*(vs[-i_est+1:]-vc)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,A,T,l


def integrate_delay_equation_kick(v0,T,g,dt=1e-3,i_kick=100,fraction_for_parameter_estimation=0.2,sigma=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        newv=vs[-1]-g*vds[-1]*dt+np.random.normal(scale=sigma*np.sqrt(dt))
        if i==i_kick:
            newv+=1
        vs.append(newv)
        vds.append(vs[max(0,len(vs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,A,T,l

def integrate_delay_equation_gaussian_initial_condition(T,g,dt=1e-3,sigma_gaussian=0.1,fraction_for_parameter_estimation=0.2,sigma=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=np.zeros(l)
    vcs=np.exp(-(tt[:l]-tt[l//2])**2/(2*sigma_gaussian**2))/(sigma_gaussian*np.sqrt(2*np.pi))
    vs=list(vs)
    vcs=list(vcs)
    for i in range(len(tt)-l):
        newv=vs[-1]+g*(vcs[len(vcs)-l]-vs[len(vs)-l])*dt+np.random.normal(scale=sigma*np.sqrt(dt))
        vs.append(newv)
        vcs.append(0)
    vs=np.array(vs)    
    vcs=np.array(vcs)    

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vcs,A,T,l

def integrate_delay_equation_bump_initial_condition(T,g,dt=1e-3,bump_width=0.1,fraction_for_parameter_estimation=0.2,sigma=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=np.zeros(l)
    vcs=3/4/bump_width*(1-(tt[:l]-tt[l//2])**2/bump_width**2)
    vcs[vcs<0]=0
    vs=list(vs)
    vcs=list(vcs)
    for i in range(len(tt)-l):
        newv=vs[-1]+g*(vcs[len(vcs)-l]-vs[len(vs)-l])*dt+np.random.normal(scale=sigma*np.sqrt(dt))
        vs.append(newv)
        vcs.append(0)
    vs=np.array(vs)    
    vcs=np.array(vcs)    

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vcs,A,T,l


def bump_kernel(t,delay,bump_width,k):
    if len(t)==0:
        if np.abs(t-delay)<bump_width:
            return k*3/4/bump_width*(1-(t-delay)**2/bump_width**2)
        else:
            return 0
    else:
        kk=k*3/4/bump_width*(1-(t-delay)**2/bump_width**2)
        kk[kk<0]=0
        return kk
    
    
def integrate_delay_equation_bump_kernel(s0,T,alpha,params,dt=1e-3,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    tk=dt*np.arange(1,round((params[0]+params[1])/dt))
    kernel=bump_kernel(tk,*params)[::-1]
    ss=[s0]
    vds=[0]
    vcs=[1]
    for i in range(len(tt)-1):
        snew=ss[-1]+vds[-1]*dt
        vcnew=0
        vs_tmp=vcs[max(0,len(vcs)-len(kernel)):]-alpha*np.array(ss[max(0,len(ss)-len(kernel)):])
        vdnew=np.sum(vs_tmp*kernel[-len(vs_tmp):]*dt)
        ss.append(snew)
        vcs.append(vcnew)
        vds.append(vdnew)
    ss=np.array(ss)    
    vds=np.array(vds)
    vcs=np.array(vcs)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,vds,vcs,A,T


def integrate_delay_equation_custom_kernel(s0,T,alpha,kernel_times,kernel_points,dt=1e-3,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    kernel=np.interp(dt*np.arange(1,round(kernel_times[-1]/dt)),kernel_times,kernel_points)[::-1]
    ss=[s0]
    vds=[0]
    vcs=[1]
    
    for i in range(len(tt)-1):
        snew=ss[-1]+vds[-1]*dt
        vcnew=0
        vs_tmp=vcs[max(0,len(vcs)-len(kernel)):]-alpha*np.array(ss[max(0,len(ss)-len(kernel)):])
        vdnew=np.sum(vs_tmp*kernel[-len(vs_tmp):]*dt)
        ss.append(snew)
        vcs.append(vcnew)
        vds.append(vdnew)
    ss=np.array(ss)    
    vds=np.array(vds)
    vcs=np.array(vcs)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,vds,vcs,A,T


def integrate_delay_equation_custom_kernel_additional_delay(T,alpha,kernel_times,kernel_points,dt=1e-3,fraction_for_parameter_estimation=0.2,tau=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    l=round(tau/dt)
    kernel=np.interp(dt*np.arange(1,round(kernel_times[-1]/dt)),kernel_times,kernel_points)[::-1]
    ss=list(np.zeros(len(kernel)+l))
    vds=list(np.zeros(len(kernel)+l))
    vcs=list(np.zeros(len(kernel)+l))
    vcs[-1]=1
    
    for i in range(len(tt)-1):
        snew=ss[-1]+vds[-1]*dt
        vcnew=0
        #vs_tmp=vcs[max(0,len(vcs)-len(kernel)):]-alpha*np.array(ss[max(0,len(ss)-len(kernel)):])
        vs_tmp=vcs[max(0,len(vcs)-len(kernel)):]-alpha*np.array(ss[max(0,len(ss)-len(kernel)-l):max(0,len(ss)-l)])
        vdnew=np.sum(vs_tmp*kernel[-len(vs_tmp):]*dt)
        ss.append(snew)
        vcs.append(vcnew)
        vds.append(vdnew)
    ss=np.array(ss)    
    vds=np.array(vds)
    vcs=np.array(vcs)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,vds,vcs,A,T

def integrate_delay_equation_custom_kernel_additional_delay_nonlinear(T,alpha,kernel_times,kernel_points,dt=1e-3,fraction_for_parameter_estimation=0.2,tau=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    l=round(tau/dt)
    kernel=np.interp(dt*np.arange(1,round(kernel_times[-1]/dt)),kernel_times,kernel_points)[::-1]
    ss=list(np.zeros(len(kernel)+l))
    vds=list(np.zeros(len(kernel)+l))
    vcs=list(np.zeros(len(kernel)+l))
    vcs[-1]=1
    
    for i in range(len(tt)-1):
        snew=ss[-1]+vds[-1]*dt
        vcnew=0
        #vs_tmp=vcs[max(0,len(vcs)-len(kernel)):]-alpha*np.array(ss[max(0,len(ss)-len(kernel)):])
        vs_tmp=vcs[max(0,len(vcs)-len(kernel)):]-alpha*np.array(ss[max(0,len(ss)-len(kernel)-l):max(0,len(ss)-l)])
        vdnew=np.sum(vs_tmp*kernel[-len(vs_tmp):]*dt)
        ss.append(snew)
        vcs.append(vcnew)
        vds.append(vdnew)
    ss=np.array(ss)    
    vds=np.array(vds)
    vcs=np.array(vcs)

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,vds,vcs,A,T


def integrate_delay_equation_sigmoid_kernel(v0,T,g,r,dt=1e-3,fraction_for_parameter_estimation=0.2,tau_rise=0.05,tau_decay=2):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    kernel=(np.exp(-dt*np.arange(1,round(3*tau_decay/dt))/tau_decay)-np.exp(-dt*np.arange(1,round(3*tau_decay/dt))/tau_rise))[::-1]
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        vs.append(vs[-1]+g*asymmetric_sigmoid(-vds[-1],ap=1,am=r,k=1)*dt)
        vs_tmp=vs[max(0,len(vs)-len(kernel)):]
        vds.append(np.sum(vs_tmp*kernel[-len(vs_tmp):])/np.sum(kernel[-len(vs_tmp):]))
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,A,T

def integrate_delay_equation_kernel(v0,T,g,dt=1e-3,fraction_for_parameter_estimation=0.2,tau_rise=0.05,tau_decay=2):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    kernel=(np.exp(-dt*np.arange(1,round(3*tau_decay/dt))/tau_decay)-np.exp(-dt*np.arange(1,round(3*tau_decay/dt))/tau_rise))[::-1]
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        vs.append(vs[-1]-g*vds[-1]*dt)
        vs_tmp=vs[max(0,len(vs)-len(kernel)):]
        vds.append(np.sum(vs_tmp*kernel[-len(vs_tmp):])/np.sum(kernel[-len(vs_tmp):]))
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,A,T


def double_peak_kernel(t,t1=-0.07,t2=-0.2,t3=-0.6):
    w1=t1-t2
    w2=t2-t3
    
    if t>t3 and t<t2:
        return -w1/w2*np.cos((t-(t2+t3)/2)*np.pi/w2)
    elif t>t2 and t<t1:
        return np.cos((t-(t1+t2)/2)*np.pi/w1)
    else:
        return 0

def integrate_equation_double_peak_kernel(s0,T,g,st=1,dt=1e-3,fraction_for_parameter_estimation=0.2,D=0,t1=-0.07,t2=-0.2,t3=-0.6,multiplicative=True):
    tt=np.linspace(0,T,int(T/dt)+1)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    kernel_times=-np.arange(round(np.abs(t3/dt)))*dt
    kernel=np.array([double_peak_kernel(t,t1=t1,t2=t2,t3=t3) for t in kernel_times[::-1]])
    ss=[s0]
    for i in range(len(tt)-1):
        ss_c=np.array(ss[max(0,len(ss)-len(kernel)):])
        integral=np.sum((ss_c-st)*kernel[-len(ss_c):])
        snew=ss[-1]-g*integral*dt
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        if multiplicative:
            snew+=eps*ss[-1]
        else:
            snew+=eps
        ss.append(snew)
        
    ss=np.array(ss)    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,A,T




def integrate_delay_equation_sigmoid_lowpass(v0,T,tau,tau_lp=0,r=1,dt=1e-3,fraction_for_parameter_estimation=0.2):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    if tau_lp==0:
        beta=0
    else:
        beta=np.exp(-dt/tau_lp)
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        vs.append(vs[-1]+asymmetric_sigmoid(-vds[-1],ap=1,am=r,k=1)*dt)
        vds.append((1-beta)*vs[max(0,len(vs)-l)]+beta*vds[-1])
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero(vs[-i_est:-1]*vs[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,A,T,l


def integrate_delay_equation_log(v0,T,g,vc=1,dt=1e-3,fraction_for_parameter_estimation=0.5):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(1/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    vs=[v0]
    vds=[v0]
    for i in range(len(tt)-1):
        vs.append(vs[-1]+vs[-1]*g*(vc-vds[-1])*dt)
        vds.append(vs[max(0,len(vs)-l)])
    vs=np.array(vs)    
    vds=np.array(vds)

    A=np.max(vs[-i_est:])-np.min(vs[-i_est:])
    zeros=np.nonzero((vs[-i_est:-1]-vc)*(vs[-i_est+1:]-vc)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,vs,vds,A,T,l


def integrate_delay_equation_gain_adaptation(s0,T,alpha,tau1,tau2,kt=0.75,lmdb=1,dt=1e-3,fraction_for_parameter_estimation=0.5):
    tt=np.linspace(0,T,int(T/dt)+1)
    l1=int(round(tau1/dt))
    l2=int(round((tau1+tau2)/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    sd1s=[s0]
    sd2s=[s0]
    ks=[kt]
    for i in range(len(tt)-1):
        #ss.append(ss[-1]-alpha*sd2s[-1]**2/sd1s[-1]*dt)
        ss.append(ss[-1]-alpha*ks[-1]*sd2s[-1]*dt)
        ks.append(ks[-1]-lmdb*(ks[-1]-kt/tau1*sd1s[-1]/(alpha*sd2s[-1]))*dt)
        sd1s.append(ss[max(0,len(ss)-l1)])
        sd2s.append(ss[max(0,len(ss)-l2)])
    ss=np.array(ss)    
    ks=np.array(ks)    
    sd1s=np.array(sd1s)    
    sd2s=np.array(sd2s) 

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero(ss[-i_est:-1]*ss[-i_est+1:]<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,sd1s,sd2s,A,T,l1,l2


def integrate_delay_equation_gain_estimate(s0,T,alpha0,vc0,vc=1,alpha=1,tau=1,k=0.75,lmdb=1,dt=1e-3,fraction_for_parameter_estimation=0.5,log=False,D=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    alphas=[alpha0]
    vcs=[vc0]
    for i in range(len(tt)-1):
        #ss.append(ss[-1]-alpha*sd2s[-1]**2/sd1s[-1]*dt)
        

        netv=vc-alpha*ss[max(0,len(ss)-l)]
        #netv=vc-alpha*np.mean(ss[max(0,len(ss)-l):len(ss)])

        if log:
            news=ss[-1]-k*ss[-1]*(ss[-1]-vcs[-1]/alphas[-1])*dt
            newalpha=alphas[-1]-lmdb*alphas[-1]*(alphas[-1]-(vcs[-1]-netv)/ss[-1])*dt
            newvc=vcs[-1]-lmdb*vcs[-1]*(vcs[-1]-(netv+alphas[-1]*ss[-1]))*dt

        else:
            news=ss[-1]-k*(ss[-1]-vcs[-1]/alphas[-1])*dt
            newalpha=alphas[-1]-lmdb*(alphas[-1]-(vcs[-1]-netv)/ss[-1])*dt
            newvc=vcs[-1]-lmdb*(vcs[-1]-(netv+alphas[-1]*ss[-1]))*dt

        
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        news+=eps
                
        
        
        alphas.append(newalpha)
        vcs.append(newvc)
        ss.append(news)
        
    ss=np.array(ss)    
    alphas=np.array(alphas)    
    vcs=np.array(vcs)  


    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero((ss[-i_est:-1]-vc/alpha)*(ss[-i_est+1:]-vc/alpha)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,alphas,vcs,A,T,l



def integrate_delay_equation_height_estimate(s0,T,h0,vc0,vc=1,h=1,tau=1,k=0.75,lmdb=1,dt=1e-3,fraction_for_parameter_estimation=0.5,log=False,D=0):
    tt=np.linspace(0,T,int(T/dt)+1)
    l=int(round(tau/dt))
    i_est=round(fraction_for_parameter_estimation*T/dt)
    ss=[s0]
    hs=[h0]
    vcs=[vc0]
    htargets=[h]
    vctargets=[vc]

    for i in range(len(tt)-1):
        #ss.append(ss[-1]-alpha*sd2s[-1]**2/sd1s[-1]*dt)
        

        netv=(vc-ss[max(0,len(ss)-l)])/h
        #netv=vc-alpha*np.mean(ss[max(0,len(ss)-l):len(ss)])

        if log:
            news=ss[-1]-k*ss[-1]*(ss[-1]-vcs[-1])*dt
            newh=hs[-1]-lmdb*hs[-1]*(hs[-1]-(vcs[-1]-ss[-1])/netv)*dt
            newvc=vcs[-1]-lmdb*vcs[-1]*(vcs[-1]-(hs[-1]*netv+ss[-1]))*dt

        else:
            news=ss[-1]-k*(ss[-1]-vcs[-1])*dt
            newh=hs[-1]-lmdb*(hs[-1]-np.abs((vcs[-1]-ss[-1])/netv))*dt
            newvc=vcs[-1]-lmdb*(vcs[-1]-(hs[-1]*netv+ss[-1]))*dt

        
        eps=np.random.normal(scale=np.sqrt(2*D*dt))
        news+=eps
                
        
        htargets.append(np.abs((vcs[-1]-ss[-1])/netv))
        vctargets.append((hs[-1]*netv+ss[-1]))
        hs.append(newh)
        vcs.append(newvc)
        ss.append(news)
        
    ss=np.array(ss)    
    hs=np.array(hs)    
    vcs=np.array(vcs)    
    htargets=np.array(htargets)    
    vctargets=np.array(vctargets)    

    A=np.max(ss[-i_est:])-np.min(ss[-i_est:])
    zeros=np.nonzero((ss[-i_est:-1]-vc)*(ss[-i_est+1:]-vc)<0)[0]
    if len(zeros)==0:
        T=np.nan
    else:
        T=np.mean(dt*(zeros[2:]-zeros[:-2]))

    return tt,ss,hs,vcs,htargets,vctargets,A,T,l