# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:45:40 2021

@author: Mostafa
"""

import matplotlib.pyplot as plt

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
        
    Psi_term=gamma*temp1*temp2
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
    
    NLI_terms={}
    
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
                    
                    E_temp=80/81*SymbolRate**2*real(E_temp)
                
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
                    
                    F_temp=16/81*SymbolRate**2*real(F_temp)
                
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
                        
                    G_temp=16/81*SymbolRate*real(G_temp)
                
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
    
    '''
    Function for single transceiver signal production
    
    Inputs:
    
        Power_dBm (float): The launch power of each channel
        
        n_sym (int): Number of transmitted symbols for detection in the receiver
        
        roll_off_factor (float): Roll-off factor of RRC shaping pulse
        
        mod_type (str, default is 'QPSK'): Modulation type of transmission
        
        probabilistic_shaping_matrix (ndarray): Probabilistic shaping of modulation, ignore it!
    '''
    
    return anything_you_like#!

#%%
def WDM_Signal(
        TransmitterSpec,
        SymbolRate,
        ChannelBandwidth,
        COI,
        n_sym
        ):
    
    '''
    Function for WDM signal production by multiplexing single transceiver outputs
    
    <<< This function runs on top of *Single_Transceiver_Output* function >>>
    
    Inputs:
    
        TransmitterSpec (dict): Dictionary of transmitter specifications, keys are wavelengths and
            values are tuples of power_dbm (launch power of single channel, float), mod_type (modulation type, str), 
            roll_off_factor (RRC shaping pulse roll-off factor, float) and ps_matrix (probabilistic shaping of modulation, str, deprecated)
            
        SymbolRate (float): Symbol rate of RRC shaping pulse
        
        ChannelBandwidth (float): WDM channel spacing
        
        COI (int): Channel ID, equal to wavelength index
        
        n_sym (int): Number of transmitted symbols for detection in the receiver
    '''
    
    return anything_you_like#!
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
    
    '''
    Function for SSFM operation in link
    
    <<< This function runs on top of *NLSE_Solver_Span* function >>>
    
    Inputs:
    
        WDM_Signal_in (ndarray): Transmitter output
        
        freq_D (ndarray): Discrete frequency for signal processing and filtering on
            signal
            
        alpha (float): Span attenuation coefficient
        
        beta2 (float): Span 2nd-order dispersion coefficient
        
        gamma (float): Span NL effect coefficient
        
        Lspan (float): Span length
        
        Nspan (float): Number of spans in the link
        
        AmpNF (float): Span amplifier noise figure
        
        ReceiverBandwidth (float): Receiver bandwidth (!) for data detection
        
        UpsamplingRatio (int): Upsampling factor of discrete signal
        
        LenStep: Fiber section length for recursive SSFM operation
    '''
    
    return anything_you_like#!
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
    Function for SSFM operation in link
    
    <<< This function runs on top of *NLSE_Solver_Span* function >>>
    
    Inputs:
    
        inputSignal_t (ndarray): Input signal to optical span
        
        freq_D (ndarray): Discrete frequency for signal processing and filtering on
            signal
            
        alpha (float): Span attenuation coefficient
        
        beta2 (float): Span 2nd-order dispersion coefficient
        
        gamma (float): Span NL effect coefficient
        
        Lspan (float): Span length
        
        LenStep: Fiber section length for recursive SSFM operation
    '''
    
    return anything_you_like#!
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
    
    ''' sum(rrc_filt_res**2) = 2*UpRatio '''
    
    SNR_Empirical = power_restored/var_error
    
    return 10*log10(SNR_Empirical)
#%%
def SNR_SSFM(
        alpha,beta2,gamma,Lspan,Nspan,AmpNF,
        LambdaList,
        power_dbm,mod_type,roll_off_factor,
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
#    LambdaList=[1,2,3]
    
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