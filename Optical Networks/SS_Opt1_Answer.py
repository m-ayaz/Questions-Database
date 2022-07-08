# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:45:40 2021

@author: Mostafa
"""

import matplotlib.pyplot as plt

#from pickle import load as pickleload

from tqdm import tqdm

from numpy import pi,exp,conj,sqrt,array,real,var,log10,floor,cos,\
kron,arange,reshape,ceil,sum,abs,cumsum,angle,imag

from numpy.random import randn,rand,uniform

from numpy.fft import fft,ifft

from commpy.filters import rrcosfilter

from scipy.signal import resample_poly,upfirdn
#%%
def RRC_f(f,SymbolRate,roll_off_factor):
    '''
    Pulse Shape; for now, it is assumed to be
    rectangular in frequency
    '''
    pass_band=abs(f)<SymbolRate*(1-roll_off_factor)/2
    
    if roll_off_factor==0:
        transient_band=0
    else:
        transient_band=cos(pi/2/roll_off_factor/SymbolRate*(abs(f)-SymbolRate*(1-roll_off_factor)/2))*\
        (abs(f)>=SymbolRate*(1-roll_off_factor)/2)*(abs(f)<SymbolRate*(1+roll_off_factor)/2)
    
    return (pass_band+transient_band)/SymbolRate
#%%
def _2DMC(func,x0,x1,y0,y1,_N=10000):
    '''2D Monte Carlo Integration'''
    _N=int(_N)
    x=uniform(x0,x1,_N)
    y=uniform(y0,y1,_N)
    return sum(func(x,y))*(x1-x0)*(y1-y0)/_N
#%%
def _3DMC(func,x0,x1,y0,y1,z0,z1,_N=10000):
    '''3D Monte Carlo Integration'''
    _N=int(_N)
    x=uniform(x0,x1,_N)
    y=uniform(y0,y1,_N)
    z=uniform(z0,z1,_N)
    return sum(func(x,y,z))*(x1-x0)*(y1-y0)*(z1-z0)/_N
#%%
def _4DMC(func,w0,w1,x0,x1,y0,y1,z0,z1,_N=10000):
    '''4D Monte Carlo Integration'''
    _N=int(_N)
    w=uniform(w0,w1,_N)
    x=uniform(x0,x1,_N)
    y=uniform(y0,y1,_N)
    z=uniform(z0,z1,_N)
    return sum(func(w,x,y,z))*(w1-w0)*(x1-x0)*(y1-y0)*(z1-z0)/_N
#%%
def _5DMC(func,v0,v1,w0,w1,x0,x1,y0,y1,z0,z1,_N=10000):
    '''5D Monte Carlo Integration'''
    _N=int(_N)
    v=uniform(v0,v1,_N)
    w=uniform(w0,w1,_N)
    x=uniform(x0,x1,_N)
    y=uniform(y0,y1,_N)
    z=uniform(z0,z1,_N)
    return sum(func(v,w,x,y,z))*(v1-v0)*(w1-w0)*(x1-x0)*(y1-y0)*(z1-z0)/_N
#%%
def Upsilon(f1,f2,f,alpha,beta2,gamma,Lspan,Nspan):
    '''
    "f1" and "f2" must have the same size i.e.
    they must be both vectors or matrices.
    
    "link_index" is the same "k" in the main model equations.
    '''
    if gamma==0:
        return 0
    
    theta_beta=4*pi**2*beta2*(f1-f)*(f2-f)
    
    if alpha<1e-6:
        raise Exception('Too small fiber attenuation')
    
    temp1=(1-exp(-alpha*Lspan+1j*Lspan*theta_beta))/(alpha-1j*theta_beta)
    
    if beta2==0:
        temp2=Nspan
    else:
        temp2=(exp(1j*Nspan*Lspan*theta_beta)-1)/(exp(1j*Lspan*theta_beta)-1)
        
    temp3=exp(-2j*pi**2*Nspan*beta2*Lspan*f**2)
    
    Psi_term=gamma*temp1*temp2*temp3
    '''==============================================='''
    return Psi_term
#%%
def SNR_EGN(alpha,beta2,gamma,Lspan,Nspan,
            LambdaList,ChannelBandwidth,SymbolRate,f_COI,
            power_dBm,ModulationType,
            NLI_terms=None,_NMC=10000,printlog=False
            ):
    
    if NLI_terms==None:
        NLI_terms=NLI_EGN(
            alpha,beta2,gamma,Lspan,Nspan,
            LambdaList,ChannelBandwidth,SymbolRate,f_COI,
            ModulationType,roll_off_factor,
            _NMC=10000,printlog=False
        )
    
    power_linear={tup: 10**(0.1*power_dBm-3) for tup in LambdaList}
    
    NLI_var_coherent=0
    G_ASE=0
    
    for nu_kap1 in LambdaList:
        for nu_kapt in LambdaList:
            for nu_kap2 in LambdaList:
                
                if nu_kap1<nu_kap2:
                    continue
                
                NLI_var_coherent=NLI_var_coherent+\
                NLI_terms[nu_kap1,nu_kapt,nu_kap2]*\
                power_linear[nu_kap1]*\
                power_linear[nu_kapt]*\
                power_linear[nu_kap2]
                
    h=6.62607004e-34
    c=299792458
    C_lambda=1.55e-6
    nu=c/C_lambda
    
    G_ASE=h*nu/2*(exp(alpha*Lspan)*10**(AmpNF/10)-1)*Nspan*2
        
    SNR_dB_coherent=power_dBm-30-10*log10(G_ASE*SymbolRate+NLI_var_coherent)
    
    return SNR_dB_coherent
#%%
def NLI_EGN(
        alpha,beta2,gamma,Lspan,Nspan,
        LambdaList,ChannelBandwidth,SymbolRate,f_COI,
        ModulationType,roll_off_factor,
        _NMC=10000,printlog=False
        ):
    
    if ModulationType=='QPSK':
        ModPhi=-1
        ModPsi=4
    elif ModulationType=='16QAM':
        ModPhi=-0.68
        ModPsi=2.08
    
    NLI_terms={}
    
#    LambdaList1=LambdaList.copy()
    
#    LambdaList=[ChannelBandwidth*x for x in LambdaList1]
#    print(LambdaList)
    
    for i in LambdaList:
        for j in LambdaList:
            for k in LambdaList:
                
                nu_kap1=ChannelBandwidth*i
                nu_kapt=ChannelBandwidth*j
                nu_kap2=ChannelBandwidth*k
                
                if nu_kap1<nu_kap2:
                    continue
                
                if printlog:
                    print('[k1 ,kt ,k2] = ['+str(nu_kap1)+', '+str(nu_kapt)+', '+str(nu_kap2)+']\n')
                
                Omega=nu_kap1-nu_kapt+nu_kap2
                
                D_temp=_3DMC(lambda f1,f2,f:
                    Upsilon(f1+nu_kap1,f2+nu_kap2,f,alpha,beta2,gamma,Lspan,Nspan)*
                    conj(Upsilon(f1+nu_kap1,f2+nu_kap2,f,alpha,beta2,gamma,Lspan,Nspan))*
                    abs(RRC_f(f1,SymbolRate,roll_off_factor))**2*
                    abs(RRC_f(f2,SymbolRate,roll_off_factor))**2*
                    abs(RRC_f(f1+f2-f+Omega,SymbolRate,roll_off_factor))**2,
                    -ChannelBandwidth/2,ChannelBandwidth/2,
                    -ChannelBandwidth/2,ChannelBandwidth/2,
                    f_COI-ChannelBandwidth/2,f_COI+ChannelBandwidth/2
                    )
                
                D_temp=16/27*SymbolRate**3*real(D_temp)
                
                E_temp=F_temp=G_temp=0
                
                if nu_kap2==nu_kapt:
                    
                    E_temp=_4DMC(lambda f1,f2,f2p,f:
                        Upsilon(f1+nu_kap1,f2+nu_kap2,f,alpha,beta2,gamma,Lspan,Nspan)*
                        conj(Upsilon(f1+nu_kap1,f2p+nu_kap2,f,alpha,beta2,gamma,Lspan,Nspan))*
                        abs(RRC_f(f1,SymbolRate,roll_off_factor))**2*
                        RRC_f(f2,SymbolRate,roll_off_factor)*
                        conj(RRC_f(f1+f2-f+Omega,SymbolRate,roll_off_factor))*
                        conj(RRC_f(f2p,SymbolRate,roll_off_factor))*
                        RRC_f(f1+f2p-f+Omega,SymbolRate,roll_off_factor),
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        f_COI-ChannelBandwidth/2,f_COI+ChannelBandwidth/2
                        )
                    
                    E_temp=80/81*SymbolRate**2*ModPhi*real(E_temp)
                
                if nu_kap1==nu_kap2:
                    
                    F_temp=_4DMC(lambda f1,f2,f1p,f:
                        Upsilon(f1+nu_kap1,f2+nu_kap2,f,alpha,beta2,gamma,Lspan,Nspan)*
                        conj(Upsilon(f1p+nu_kap1,f1+f2-f1p+nu_kap2,f,alpha,beta2,gamma,Lspan,Nspan))*
                        abs(RRC_f(f1+f2-f+Omega,SymbolRate,roll_off_factor))**2*
                        RRC_f(f1,SymbolRate,roll_off_factor)*
                        conj(RRC_f(f1+f2-f1p,SymbolRate,roll_off_factor))*
                        conj(RRC_f(f1p,SymbolRate,roll_off_factor))*
                        RRC_f(f2,SymbolRate,roll_off_factor),
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        f_COI-ChannelBandwidth/2,f_COI+ChannelBandwidth/2
                        )
                    
                    F_temp=16/81*SymbolRate**2*ModPhi*real(F_temp)
                
                if nu_kap1==nu_kap2==nu_kapt:
                    
                    G_temp=_5DMC(lambda f1,f2,f1p,f2p,f:
                        Upsilon(f1+nu_kap1,f2+nu_kap2,f,alpha,beta2,gamma,Lspan,Nspan)*
                        conj(Upsilon(f1p+nu_kap1,f2p+nu_kap2,f,alpha,beta2,gamma,Lspan,Nspan))*
                        RRC_f(f1,SymbolRate,roll_off_factor)*
                        RRC_f(f2,SymbolRate,roll_off_factor)*
                        conj(RRC_f(f1p,SymbolRate,roll_off_factor))*
                        conj(RRC_f(f2p,SymbolRate,roll_off_factor))*
                        RRC_f(f1p+f2p-f+nu_kap1,SymbolRate,roll_off_factor)*
                        conj(RRC_f(f1+f2-f+nu_kap1,SymbolRate,roll_off_factor)),
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        -ChannelBandwidth/2,ChannelBandwidth/2,
                        f_COI-ChannelBandwidth/2,f_COI+ChannelBandwidth/2
                        )
                        
                    G_temp=16/81*SymbolRate*ModPsi*real(G_temp)
                
                NLI_terms[i,j,k]=(D_temp+E_temp+F_temp+G_temp)*((nu_kap1!=nu_kap2)+1)
                
    return NLI_terms
#%%
def scatplot(x,col='k'):
    plt.figure()
    plt.plot(real(x),imag(x),col+'.')
#%%
#################################
#################################
#################################
#################################
#################################
#################################
'''Beginning of SSFM'''
#%%
def Single_Transceiver_Output(
        Power_dBm,
        n_sym,
        roll_off_factor,
        mod_type='QPSK',
        probabilistic_shaping_matrix=None
        ):
    
    Power=10**(Power_dBm*0.1-3)
    
    '''
    Power is considered as dual polarized, so that each polarization share
    is half the power value.
    '''
    if roll_off_factor<0.1:
        HALFTRUNCLENGTH=1000
    else:
        HALFTRUNCLENGTH=50
    _,rrc_filter_temp=rrcosfilter(2*HALFTRUNCLENGTH+2,roll_off_factor,1,2)
    rrc_filter=rrc_filter_temp[1:]
    
    if mod_type=='QPSK':
        mod_num_points=4
        if probabilistic_shaping_matrix==None:
            probabilistic_shaping_matrix=[[.25]*2]*2
        alphabet=array([[-1,1]])+array([[1,-1]]).T*1j
        
    elif mod_type=='16QAM':
        mod_num_points=16
        if probabilistic_shaping_matrix==None:
            probabilistic_shaping_matrix=[[1/16]*4]*4
        alphabet=array([[-3,-1,1,3]])+array([[3,1,-1,-3]]).T*1j
        
    elif mod_type=='64QAM':
        mod_num_points=64
        if probabilistic_shaping_matrix==None:
            probabilistic_shaping_matrix=[[1/64]*8]*8
        alphabet=array([[-7,-5,-3,-1,1,3,5,7]])+array([[7,5,3,1,-1,-3,-5,-7]]).T*1j
        
    else:
        raise Exception('Undefined modulation type; valid types: QPSK, 16QAM, 64QAM')
    
    alphabet=reshape(alphabet,[1,mod_num_points])[0]
    
    probabilistic_shaping_matrix=reshape(probabilistic_shaping_matrix,[1,mod_num_points])[0]
    cum_probabilistic_shaping_matrix=cumsum([0]+list(probabilistic_shaping_matrix))[:-1]
    mod_power=2*sum(abs(alphabet)**2*probabilistic_shaping_matrix)
    ind_x=sum(rand(n_sym,1)<cum_probabilistic_shaping_matrix,1)
    ind_y=sum(rand(n_sym,1)<cum_probabilistic_shaping_matrix,1)
    TxSyms=array([alphabet[ind_x],alphabet[ind_y]])*sqrt(Power/mod_power)
    TxSyms_resampled=kron(TxSyms,[1,0])
    baseband=upfirdn(rrc_filter,TxSyms_resampled)
    
    baseband=array([
            baseband[0][HALFTRUNCLENGTH:-HALFTRUNCLENGTH],
            baseband[1][HALFTRUNCLENGTH:-HALFTRUNCLENGTH]
            ])
    
    return baseband,TxSyms,rrc_filter

#%%
def WDM_Signal(TransmitterSpec,SymbolRate,ChannelBandwidth,COI,n_sym):
    
    LambdaArray=TransmitterSpec.keys()
    F_MIN=(min(LambdaArray)-0.5)*ChannelBandwidth
    
    fsampling=(max(LambdaArray)+0.5)*ChannelBandwidth-F_MIN
    UpSamplingRatio=ceil(fsampling/2/SymbolRate)+3
    
    fsampling=2*SymbolRate*UpSamplingRatio
    
    lenSignal=2*n_sym*UpSamplingRatio
    
    time_D=arange(2*n_sym*UpSamplingRatio)/fsampling
    
    Tx_Symbols={}
    
    freq_D_hat=arange(lenSignal)/lenSignal-F_MIN/fsampling
    freq_D=fsampling*(freq_D_hat-floor(freq_D_hat))+F_MIN
    
    WDM_Signal_out=0
    
    for wavelength in LambdaArray:
        
        power_dbm,mod_type,roll_off_factor,ps_matrix=TransmitterSpec[wavelength]
        
        baseband,txsyms,rrc_filter=Single_Transceiver_Output(
        power_dbm,
        n_sym,
        roll_off_factor,
        mod_type,
        probabilistic_shaping_matrix=ps_matrix
        )
        
        Tx_Symbols[wavelength]=txsyms
        
        baseband_resampled=resample_poly(baseband.T,up=UpSamplingRatio,down=1).T
        
        WDM_Signal_out+=baseband_resampled*exp(2j*pi*wavelength*ChannelBandwidth*time_D)
    
    return WDM_Signal_out,Tx_Symbols,freq_D,time_D,fsampling,UpSamplingRatio,rrc_filter
#%%
def Link_SSFM_Simulator(
        WDM_Signal_in,
        freq_D,
        alpha,
        beta2,
        gamma,
        Lspan,
        Nspan,
        AmpNF,
        ReceiverBandWidth,
        UpsamplingRatio,
        LenStep
        ):
    
    inputSignal_t_plus_NLI_ASE=WDM_Signal_in
    
    for spanind in range(Nspan):
        
        print('......... span {} of {}'.format(spanind+1,Nspan))
        
        inputSignal_f_plus_NLI=NLSE_Solver_Span(
                inputSignal_t_plus_NLI_ASE,
                freq_D,
                alpha,
                beta2,
                gamma,
                Lspan,
                LenStep
                )
        
        '''Adding ASE noise to Signal'''
        AmplifierNoiseFactor=10**(AmpNF/10)
        
        AmplifierGain=exp(alpha*Lspan/2)
        
        h=6.62607004e-34
        c=299792458
        C_lambda=1.55e-6
        nu=c/C_lambda
        
        '''ASE Variance based on exact mathematical formula'''
        N=len(inputSignal_f_plus_NLI[0])
        
        ASE_Variance=max([h*nu*(AmplifierNoiseFactor*AmplifierGain**2-1)*ReceiverBandWidth/2,0])
        
        ASE_Variance=ASE_Variance*UpsamplingRatio*2
        
        noise_process=sqrt(ASE_Variance/2)*randn(2,N)+1j*sqrt(ASE_Variance/2)*randn(2,N)
        
        inputSignal_t_plus_NLI_ASE=AmplifierGain*fft(inputSignal_f_plus_NLI)+noise_process
    
    AccDispersion=Nspan*Lspan*beta2
    
    return inputSignal_t_plus_NLI_ASE,AccDispersion
#%%
def NLSE_Solver_Span(
        inputSignal_t,
        freq_D,
        alpha,
        beta2,
        gamma,
        Lspan,
        LenStep
        ):
    '''
    NLSE Solver in Span
    
    "inputSignal_f" is the frequency-domain Electrical Field, which will
    be returned as output after affected by impairments
    
    "freq_D" is the discrete frequency generated alongside the signal
    '''
    
    if beta2!=0 and gamma!=0:
        NumofFiberSections=int(ceil(Lspan/LenStep))
    else:
        NumofFiberSections=1
    
    h=Lspan/NumofFiberSections
    
    inputSignal_f=ifft(inputSignal_t)
    
    # Linear Impairments, Full and half section
    Full_Linear_Exp=exp(-alpha*h/2+2j*beta2*h*pi**2*freq_D**2)
    Half_Linear_Exp=exp(-alpha*h/4+1j*beta2*h*pi**2*freq_D**2)
    
    ''' First half step '''
    inputSignal_f_1=inputSignal_f*Half_Linear_Exp
    
    if alpha==0:
        h_eff=h
    else:
        h_eff=(1-exp(-alpha*h))/alpha
    
    '''Solver core'''
    for i in tqdm(range(NumofFiberSections),position=0,leave=True):
        '''At each step: First Nonlinear, then linear'''
        inputSignal_f_1_inv=fft(inputSignal_f_1)
        inputSignal_f_2=inputSignal_f_1_inv*exp(1j*h_eff*gamma*sum(abs(inputSignal_f_1_inv)**2,0)*8/9)
        inputSignal_f_1=ifft(inputSignal_f_2)*Full_Linear_Exp
        
    '''Half step linear backwards'''
    inputSignal_f_plus_NLI=inputSignal_f_1/Half_Linear_Exp
    
    return inputSignal_f_plus_NLI
#%%
def EDC(ReceivedSignal,freq_D,AccDispersion):
    '''Electronic Dispersion Compensation'''
    return fft(ifft(ReceivedSignal)*exp(-2j*pi**2*freq_D**2*AccDispersion))
#%%
def to_baseband(ReceivedSignal,time_D,Wavelength):
    return ReceivedSignal*exp(-2j*pi*Wavelength*time_D)
#%%
def AnalogLowPassFilter(ReceivedSignal,fsampling,ChannelBandwidth):
    '''Ideal LPF'''
    passband_freq=ChannelBandwidth
    sig_length=len(ReceivedSignal[0])
    N_ones=int(passband_freq/fsampling*sig_length/2)
    filt_freq=[1]*N_ones+[0]*(sig_length-2*N_ones)+[1]*N_ones
    output=ifft(fft(ReceivedSignal)*filt_freq)
    return output
#%%    
def MF_DownSample(ReceivedSignal,dnsampling_ratio,rrc_filter,roll_off_factor):
    '''Constructing UpSampled Matched Filter'''
    N_FILT_RES=int(len(rrc_filter)/2-0.5)*dnsampling_ratio*2+2
    
    _,rrc_filt_res=rrcosfilter(int(N_FILT_RES),roll_off_factor,1,2*dnsampling_ratio)
    
    rrc_filt_res=rrc_filt_res[1:]
    
    output=upfirdn(rrc_filt_res,ReceivedSignal)
    HALF_N_FILT_RES=int(N_FILT_RES/2)-1
    output=[
        output[0][HALF_N_FILT_RES:-HALF_N_FILT_RES],
        output[1][HALF_N_FILT_RES:-HALF_N_FILT_RES]
        ]
    output=resample_poly(output,up=1,down=dnsampling_ratio,axis=1)
    return output
#%%
def SymbolDetector(ReceivedSignal,TotalNumOfSymbols):
    '''Extracting received symbols from the received signal'''
    '''Regularly picking symbols alternatively'''
    Detected_ModulationSymbols=array([
            ReceivedSignal[0][:(TotalNumOfSymbols-1)*2+1:2],
            ReceivedSignal[1][:(TotalNumOfSymbols-1)*2+1:2]
            ])
    return Detected_ModulationSymbols
#%%
def PhaseRotationCompensator(RxModulationSymbols,TxModulationSymbols):
    '''The optimum phase rotation angle is that calculated through xy* where
    "x" denotes transmitted and "y" denoted received.'''
    opt_angle=array([
            angle(sum(TxModulationSymbols*conj(RxModulationSymbols),1))
            ]).T
    return RxModulationSymbols*exp(1j*opt_angle)
#%%
'''Time to detect!'''
def Receiver_Detector(
        Receiver_Input,
        freq_D,
        time_D,
        Wavelength,
        SymbolRate,
        ChannelBandwidth,
        fsampling,
        TxSyms,
        n_sym,
        rrc_filter,
        UpSamplingRatio,
        AccDispersion
        ):
    
    del_margin_percent=0.3
    
    ''' Removing dispersion '''
    Total_Signal_at_EDC_output=EDC(Receiver_Input,freq_D,AccDispersion)
    
    ''' Bringing CUT to baseband '''
    Total_Signal_to_baseband=to_baseband(Total_Signal_at_EDC_output,time_D,Wavelength)
    
    ''' Picking CUT by passing through analog low-pass filter '''
    Total_Signal_to_baseband_filtered=AnalogLowPassFilter(
            Total_Signal_to_baseband,
            fsampling,
            ChannelBandwidth
            )
    
    ''' Downsampling the analog low-pass filter output for detecting symbols '''
    Total_Signal_to_baseband_filtered_DownSampled=MF_DownSample(
            Total_Signal_to_baseband_filtered,
            UpSamplingRatio,
            rrc_filter,
            roll_off_factor
            )
    
    ''' Detecting noisy symbols '''
    RxSyms=SymbolDetector(
            Total_Signal_to_baseband_filtered_DownSampled,
            n_sym
            )
    
    ''' Detected noisy symbols phase correction by means of the transmitted symbols '''
    RxSyms_rot=PhaseRotationCompensator(RxSyms,TxSyms)
    
    del_margin=int(del_margin_percent*len(TxSyms[0])/2)
    
    CUTLightPathTxSymbols=array([
            TxSyms[0][del_margin:-del_margin],
            TxSyms[1][del_margin:-del_margin]
            ])
    
    RxModulationSymbols_rot=array([
            RxSyms_rot[0][del_margin:-del_margin],
            RxSyms_rot[1][del_margin:-del_margin]
            ])
    
    TxSyms_x,TxSyms_y=CUTLightPathTxSymbols
    RxSyms_x_rot,RxSyms_y_rot=RxModulationSymbols_rot
    
    ran_gain_restored_x=real(sum(TxSyms_x*conj(RxSyms_x_rot))/sum(abs(TxSyms_x)**2))
    ran_gain_restored_y=real(sum(TxSyms_y*conj(RxSyms_y_rot))/sum(abs(TxSyms_y)**2))
    
    ran_gain_restored=(ran_gain_restored_y+ran_gain_restored_x)/2
    
    RxSyms=array([RxSyms_x_rot,RxSyms_y_rot])
    
    TxSyms=array([TxSyms_x,TxSyms_y])
    
    error=TxSyms-RxSyms/ran_gain_restored
    
    var_error=sum(var(error,1))
    
    power_restored=sum(var(RxSyms,1))/ran_gain_restored**2-var_error
    
#    print(power_restored)
    
    ''' sum(rrc_filt_res**2) = 2*UpRatio '''
    
    SNR_Empirical = power_restored/var_error
    
    return 10*log10(SNR_Empirical)
#%%
def SNR_SSFM(
        alpha,beta2,gamma,Lspan,Nspan,AmpNF,
        LambdaList,
        power_dbm,mod_type,roll_off_factor,
#        TransmitterSpec,
        SymbolRate,
        ChannelBandwidth,
        COI,
        n_sym,
        ps_matrix=None
        ):
    
    TransmitterSpec={wavelength:(power_dbm,mod_type,roll_off_factor,ps_matrix) for wavelength in LambdaList}
    
    WDM_Signal_out,Tx_Symbols,freq_D,time_D,fsampling,UpSamplingRatio,rrc_filter=WDM_Signal(
            TransmitterSpec,
            SymbolRate,
            ChannelBandwidth,
            COI,
            n_sym
            )
    
    Link_Signal_out,AccDispersion=Link_SSFM_Simulator(
        WDM_Signal_out,
        freq_D,
        alpha,
        beta2,
        gamma,
        Lspan,
        Nspan,
        AmpNF,
        SymbolRate*(1+TransmitterSpec[COI][2]*0),
        UpSamplingRatio,
        LenStep
        )
    
    return Receiver_Detector(
        Link_Signal_out,
        freq_D,
        time_D,
        COI*ChannelBandwidth,
        SymbolRate,
        ChannelBandwidth,
        fsampling,
        Tx_Symbols[COI],
        n_sym,
        rrc_filter,
        UpSamplingRatio,
        AccDispersion
        )
#%%
if __name__=='__main__':
    
    plt.close('all')
    
    alpha=4.61e-5
    beta2=-21e-27
    gamma=1.3e-3*0
    AmpNF=5
    
    Lspan=100e3
    Nspan=5
    
    mod_type,roll_off_factor,ps_matrix='QPSK',0.2,None
    
    SymbolRate=32e9
    ChannelBandwidth=38.5e9
    
    n_sym=500
    LenStep=100
    
    COI=1
    
    PowerSet_dBm=arange(-5,6,1)
    
    LambdaList=[1]
#    LambdaList=[1,3,6,10]
    
    SNR_SSFM_dB=[]
    SNR_EGN_dB=[]
    
    for power_dbm in PowerSet_dBm:
        
        print('............power (dbm) = {}'.format(power_dbm))
    
        tempSNR=SNR_SSFM(
                alpha,beta2,gamma,Lspan,Nspan,AmpNF,
                LambdaList,
                power_dbm,mod_type,roll_off_factor,
                SymbolRate,
                ChannelBandwidth,
                COI,
                n_sym,
                ps_matrix=None
                )
        
        SNR_SSFM_dB.append(tempSNR)
        
        SNR_EGN_dB.append(
                SNR_EGN(alpha,beta2,gamma,Lspan,Nspan,
                    LambdaList,ChannelBandwidth,SymbolRate,1*ChannelBandwidth,
                    power_dbm,mod_type,
                    NLI_terms=None,_NMC=10000,printlog=True
                    )
                )
    #%%
    plt.plot(PowerSet_dBm,SNR_EGN_dB)
    plt.plot(PowerSet_dBm,SNR_SSFM_dB,'.')
    plt.grid('on')