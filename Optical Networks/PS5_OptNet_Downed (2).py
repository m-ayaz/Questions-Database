# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:43:53 2020

@author: Glory
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:13:41 2020

@author: Glory
"""

import numpy as np
import matplotlib.pyplot as plt

exp=np.exp
array=np.array
pi=np.pi
arcsinh=np.arcsinh

'''
[
 [Span1_Ch1,Span1_Ch2],
 [Span2_Ch1,Span2_Ch2]
]

e.g. G_bar[2][3] refers to *G_bar* at the 3rd span and the 4th channel
'''
#%%
def toarr(x):
    try:
        return array([array(ind) for ind in x])
    except TypeError:
        return x
#%%    
def Gamma(alphaSpan,LengthSpan):
    '''
    The power-gain/loss at frequency f due to lumped
    elements, such as amplifiers and gain-flattening
    filters (GFFs), placed at the end of the span fiber...
    
    *This function operates on a single span and its return type is a float
    '''
#    alphaSpan=toarr(alphaSpan)
#    LengthSpan=toarr(LengthSpan)
    
#    NumSpan=len(alphaSpanvec)
    alphaSpan=array(alphaSpan)
    LengthSpan=array(LengthSpan)
    
    return exp(2*alphaSpan*LengthSpan)
#%%
def alpha(f_comb):
    '''
    The span attenuation vector...
    
    *This function operates on a single span and its return type is a vector
    of length NumChannels per span*
    '''
    return toarr([0.22/4.343*1e-3/2]*len(f_comb))

#def beta2(f,NumSpan,MaxNumofChannels):
#    return
#%%
def beta2_bar(f_comb,f_CUT,beta2,beta3,f_c):
    '''
    Eq [5]
    
    The frequency f_c is where β_2 and β_3
    are calculated in the n-th span (float type)
    
    *f_CUT* : float
    
    *f_c*   : float
    
    *β_2*   : float
    
    *β_3*   : float
    
    f_comb is a matrix of length **NumSpan**, each of its elements being
    a list containing set of wavelengths inside each span
    
    *This function operates on a single span and its return type is a vector
    of length NumChannels per span*
    '''
#    beta2=toarr(beta2)
#    beta3=toarr(beta3)
    
#    f_span=toarr(f_span)
#    f_c=toarr(f_c)
    
    return beta2+pi*beta3*array(f_comb+f_CUT-2*f_c)

#def I_CUT(f_CUT,f_c,R_CUT,beta2,beta3):
#    '''
#    Eq [3]
#    '''
#    temp1=1/2/pi/abs(beta2_bar(f_CUT,f_CUT,f_c,beta2,beta3))/2/alpha(f_CUT,NumSpan)
#    temp2=arcsinh(pi**2/2*abs(beta2_bar(f_CUT,f_CUT,f_c,beta2,beta3)/2/alpha(f_CUT,NumSpan))*R_CUT**2)
#    return temp1*temp2
#%%
def I(f_comb,R_comb,f_CUT,R_CUT,beta2,beta3,f_c=0):
    '''
    I at each span
    Eqs [3,4]
    
    *This function operates on a single span and its return type is a vector
    of length NumChannels per span*
    '''
#    beta2=toarr(beta2)
#    beta3=toarr(beta3)
    
    f_comb=toarr(f_comb)
    
#    print(f_comb)
#    f_c=toarr(f_c)
    
    R_comb=toarr(R_comb)
    
#    temp1=[]
#    temp2=[]
    
#    for i in range(len(beta2)):
    temp1=arcsinh((pi**2)*abs(beta2_bar(f_comb,f_CUT,beta2,beta3,f_c)/2/alpha(f_comb))*(f_comb-f_CUT+R_comb/2)*R_CUT)
    temp2=arcsinh((pi**2)*abs(beta2_bar(f_comb,f_CUT,beta2,beta3,f_c)/2/alpha(f_comb))*(f_comb-f_CUT-R_comb/2)*R_CUT)
    
#    print(temp1)
#    print(temp2)
    
#    return alpha(f_comb,NumSpan)
    return (temp1-temp2)/(4*pi*abs(beta2_bar(f_comb,f_CUT,beta2,beta3,f_c))*2*alpha(f_comb))
#%%
def beta2_bar_acc(f_comb,f_CUT,indSpan,n_ch,LengthSpanvec,beta2,beta3,fc=0):
    
    temp=0
    
    for k in range(indSpan-1):
        temp=temp+beta2_bar(f_comb[n_ch],f_CUT,beta2,beta3,fc)*LengthSpanvec[k]
        
    return temp
#%%
def rho(f_comb,CUTindex,R_CUT,indSpan,Phi,beta2Span,beta3Span,model_kind='CFM2'):
    '''
    rho at each span
    
    *f_comb* is the set of freqs in span
    
    *This function operates on a single span and its return type is a vector
    of length NumChannels per span*
    '''
    a1=+9.3143e-1; a10=-1.88380e0
    a2=-7.7122e-1; a11=+6.2974e-1
    a3=+9.1090e-1; a12=-1.1421e+1
    a4=-1.4555e+1; a13=+6.7368e-1
    a5=+8.5816e-1; a14=-1.17590e0
    a6=-9.9415e-1; a15=+6.4482e-3
    a7=+1.08120e0; a16=+1.8738e+5
    a8=+5.2247e-3; a17=+1.9527e+3
    a9=+9.9313e-1; a18=-2.00160e0
    '''
    Data copied from the lecture
    '''
    
#    f_CUT=f_comb[CUTindex]
    if model_kind=='CFM2':
    
        temp_rho=[]
        
        for freqind in range(len(f_comb)):
#            print(freqind)
            if freqind==CUTindex:
                temp1=a9+a10*Phi[freqind]**a11+a12*Phi[freqind]**a13*(1+a14*R_CUT**a15+a16*(abs(beta2_bar_acc(f_comb,indSpan,freqind,LengthSpanvec,beta2Span,beta3Span))+a17)**a18)
            else:
                temp1=a1+a2*Phi[freqind]**a3+a4*Phi[freqind]**a5*(1+a6*(abs(beta2_bar_acc(f_comb,indSpan,freqind,LengthSpanvec,beta2Span,beta3Span))+a7)**a8)
            
#            print(freqind)
            
            temp_rho.append(temp1)
        
#        print(toarr(temp_rho))
        return toarr(temp_rho)
    
    elif model_kind=='CFM1':
        return toarr([1]*len(f_comb))
    
    else:
        assert 1==2,'Unknown model type'
#%%
def G_NLI(f_comb,R_comb,CUTindex,Power_dBmvec,alphaSpanvec,beta2Spanvec,gammaSpanvec,LengthSpanvec,PhiSpanvec):
    '''
    *gammaSpan*  :  the NLI coefficient
    
    *LengthSpan* :  the span length
    
    *GammaSpan*  :  the span amplifer power profile
    
    *alphaSpan*  :  the span attenuation coefficient
    
    *G_bar_n*    :  G_bar in Eq [6] at the n-th span
    
    *rho_n*      :  the rho in Eq [2] (len = NumChannelsperSpan)
    
    *I_n*        :  the I in Eq [2] (len = NumChannelsperSpan)
    
    *This function operates on a single span and its return type is a float
    '''
    f_CUT=f_comb[CUTindex]
    R_CUT=R_comb[CUTindex]
    
    G_NLI_Rx_f_CUT=0
    
    Powervec=10**(0.1*toarr(Power_dBmvec)-3)
    
    for indSpan in range(len(alphaSpanvec)):
            
#        f_comb_n=f_comb[indSpan]
#        R_comb_n=R_comb[indSpan]
#        CUTindex=CUT_comb[indSpan]
        
        f_comb_n=array(f_comb)
        R_comb_n=array(R_comb)
#        CUTindex=CUT_comb[indSpan]
        
        alphaSpan=alphaSpanvec[indSpan]
        gammaSpan=gammaSpanvec[indSpan]
        beta2Span=beta2Spanvec[indSpan]
#        beta3Span=beta3Spanvec[indSpan]
        beta3Span=0
        LengthSpan=LengthSpanvec[indSpan]
        GammaSpan=Gamma(alphaSpan,LengthSpan)
        
#        PhiSpan=PhiSpanvec[indSpan]
        PhiSpan=PhiSpanvec
        
#        print(alphaSpan)
#        print(gammaSpan)
#        print(beta2Span)
#        print(beta3Span)
#        print(LengthSpan)
#        print(GammaSpan)
        
        G_bar_n=Powervec/R_comb_n
        
#        print(G_bar_n)
        
#        print(G_bar_n)
        
#            rho_n=rho(NumChannelsvec[indSpan])
        rho_n=rho(f_comb_n,CUTindex,R_CUT,indSpan,PhiSpan,beta2Span,beta3Span,model_kind='CFM1')
        
#        print(rho_n)
        
#        print(rho_n)
#            print('asasas',f_comb_n)
        
        I_n=I(f_comb_n,R_comb_n,f_CUT,R_CUT,beta2Span,beta3Span,0)
        
#        print(I_n)
        temp1=16/27*gammaSpan**2*GammaSpan*exp(-2*alphaSpan*LengthSpan)*G_bar_n[CUTindex]
        temp2=rho_n*(G_bar_n)**2*I_n*(1+(f_comb_n!=f_CUT))
        
#        print(f_comb_n/32.5e9)
#        print(1+(f_comb_n!=f_CUT))
#        print(len(temp2))
#        print(temp2)
        
#        print(G_bar_n)
#        print(rho_n)
#        print(I_n)
        
        temp3=sum(temp2)
        
#        print(temp3)
        
#        G_NLI_Rx_f_CUT=G_NLI_Rx_f_CUT+G_NLI(f_comb_n,CUTindex,gammaSpan,LengthSpan,GammaSpan,alphaSpan,G_bar_n,rho_n,I_n)
        G_NLI_Rx_f_CUT=G_NLI_Rx_f_CUT+temp1*temp3
        
#        print(temp1*temp3)
#    G_bar_CUT=G_bar[:,CUT]
    
#    rho_CUT=rho[:,CUT]
    
#    I_CUT=I[:,CUT]
#    f_CUT=f_comb_n[CUTindex]
    
#    print(GammaSpan)
    
#    print(type(gammaSpan))
#    print(len(alphaSpan))
#    print(type(LengthSpan))
#    print(type(temp3))
#    print(len(temp3))
    
    return G_NLI_Rx_f_CUT
#%%
def ASE_PSD(AmplifierNFvec,alphaSpanvec,LengthSpanvec,AmplifierGainvec=1,iscompensated=True):
    
    h=6.62607004e-34
    c=299792458
    C_lambda=1.53e-6
    nu=c/C_lambda
    
    AmplifierNFvec=array(AmplifierNFvec)
    
    AmplifierNoiseFactorvec=10**(AmplifierNFvec/10)
    
    if iscompensated==True:
        AmplifierGainvec=Gamma(alphaSpanvec,LengthSpanvec)
    else:
        AmplifierGainvec=array(AmplifierGainvec)
        
    
#    print(AmplifierNoiseFactor)
    
    return h*nu/2*sum(AmplifierNoiseFactorvec*AmplifierGainvec-1)
#%%
if __name__=='__main__':
    
    ############################################
    ############################################
    ############ Start Coding Here! ############
    ############################################
    ############################################
    
#    CUT=2
    NumSpan=1
#    NumofChannels=5
    
    CUTindex=0
    
    R=32.5e9
#    f_CUT=R*2
    
    
    #Power=array([array([1,2]),array([2,3,4])])
    #ChannelSpacing=array([array([3,4]),array([5,6,4])])
    
    #rho=[[],[]]
    
    #I=[[],[]]
    #alphaSpanvec=alpha(f_CUT,NumSpan)
    alphaSpanvec=[0.22/4.343*1e-3/2]*NumSpan
    beta2Spanvec=[-1.9378e-26]*NumSpan
#    beta3Spanvec=[0]*NumSpan
    gammaSpanvec=[1.3e-3]*NumSpan
    LengthSpanvec=[140e3]*NumSpan
    
    PhiSpanvec=[1]*1
    
    AmplifierNFvec=[5.5]*NumSpan
    
    PowerSet_dBm=np.linspace(-5,10,16)
#    Lambdavec=[1,3,6,8]
    SNR_dB_CFM=[]
#    f_comb=[[1*R,2*R,3*R,4*R,5*R],[1*R,2*R,3*R],[1*R,2*R,3*R,4*R]]
#    R_comb=[[R,R,R,R,R],[R,R,R],[R,R,R,R]]
#    CUT_comb=[1,1,1] # Second index at each comb is CUT
#    R_CUT=R
#    NumChannelsvec=[5,3,4]
    
#    f_comb=toarr([np.linspace(1,NumofChannels,NumofChannels)*R]*NumSpan)
#    f_comb=toarr([[1*R,2*R,3*R]]*NumSpan)
#    f_comb=toarr([[1,2,3,4,5]]*25+[[1,2,3]]*25)
    f_comb=[1*R]
    R_comb=[R]
    
    ASE_var=ASE_PSD(AmplifierNFvec,alphaSpanvec,LengthSpanvec)
    
    for power in PowerSet_dBm:
    
        nli=G_NLI(f_comb,R_comb,CUTindex,[power]*1,alphaSpanvec,beta2Spanvec,gammaSpanvec,LengthSpanvec,PhiSpanvec)
        
        fff=nli+ASE_var
        
        snr_temp=power-10*np.log10(fff*R)-30
        
        SNR_dB_CFM.append(snr_temp)
#    f_comb=toarr([[10*R]]*NumSpan)
#    f_comb=toarr([[1*R]]*NumSpan)
#    R_comb=toarr([[R,R,R,R,R]]*25+[[R,R,R]]*25)
#    R_comb=toarr([[R,R,R,R,R]]*25+[[R,R,R,R,R]]*25)
#    CUT_comb=toarr([1]*25+[1]*25) # Second index at each comb is CUT
#    CUT_comb=toarr([1]*50) # Second index at each comb is CUT
#    R_CUT=R
#    NumChannelsvec=toarr([NumofChannels]*NumSpan)
    
#    PhiSpanvec=toarr([[1,1,1,1,1]]*25+[[1,1,1,1,1]]*25)
#    PhiSpanvec=toarr([[1,1,1,1,1]]*25+[[1,1,1]]*25)
    
    
    
#    f_comb=[[1*R,2*R,3*R,4*R,5*R]]
#    Power_comb=[[1e-2]*5]
#    R_comb=[[R,R,R,R,R]]
#    CUT_comb=[1] # Second index at each comb is CUT
#    R_CUT=R
    
    
#    for power_dBm in PowerSet_dBm:
#        
#        power_linear=10**(0.1*power_dBm-3)
#        
##        Power_comb=[[power_linear]*5,[power_linear]*3,[power_linear]*4]
#        Power_comb=[[power_linear]*5]*25+[[power_linear]*3]*25
##        Power_comb=[[power_linear]*5]*NumSpan
#    
#        G_bar=toarr(Power_comb)/toarr(R_comb)
#        
#        G_NLI_Rx_f_CUT=0
#        
#        for indSpan in range(len(f_comb)):
#            
#            f_comb_n=f_comb[indSpan]
#            R_comb_n=R_comb[indSpan]
#            CUTindex=CUT_comb[indSpan]
#            
#            alphaSpan=alpha(f_comb_n)
#            gammaSpan=gammaSpanvec[indSpan]
#            beta2Span=beta2Spanvec[indSpan]
#            beta3Span=beta3Spanvec[indSpan]
#            LengthSpan=LengthSpanvec[indSpan]
#            GammaSpan=Gamma(alphaSpan[CUTindex],LengthSpan)
#            
#            PhiSpan=PhiSpanvec[indSpan]
#            
#    #        print(alphaSpan)
#    #        print(gammaSpan)
#    #        print(beta2Span)
#    #        print(beta3Span)
#    #        print(LengthSpan)
#    #        print(GammaSpan)
#            
#            G_bar_n=G_bar[indSpan]
#            
#    #        print(G_bar_n)
#            
##            rho_n=rho(NumChannelsvec[indSpan])
#            rho_n=rho(f_comb_n,CUTindex,R_CUT,indSpan,PhiSpan,beta2Span,beta3Span,model_kind='CFM1')
#            
##            print(rho_n)
#            
#    #        print(rho_n)
##            print('asasas',f_comb_n)
#            
#            I_n=I(f_comb_n,R_comb_n,f_CUT,R_CUT,beta2Span,beta3Span,0)
#            
#    #        print(I_n)
#            
#            G_NLI_Rx_f_CUT=G_NLI_Rx_f_CUT+G_NLI(f_comb_n,CUTindex,gammaSpan,LengthSpan,GammaSpan,alphaSpan,G_bar_n,rho_n,I_n)
#            
#        P_Noise=(G_NLI_Rx_f_CUT+ASE_PSD(len(f_comb),GammaSpan,AmplifierNF))*R_CUT
#        
#        SNR_dB_CFM.append(power_dBm-30-10*np.log10(P_Noise))
#        
    plt.plot(PowerSet_dBm,SNR_dB_CFM)
#    
##    file=open('')
    plt.grid('on')