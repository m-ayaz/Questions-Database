# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:40:43 2020

@author: Mostafa
"""

import numpy as np
#from multiprocessing import Pool
import time
from scipy import signal
import commpy as cp
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.signal import remez

rnd=np.random.rand
sin=np.sin
cos=np.cos
pi=np.pi
log=np.log
exp=np.exp
conj=np.conjugate
tan=np.tan
uniform=np.random.uniform
sqrt=np.sqrt
array=np.array
real=np.real
imag=np.imag
pi=np.pi
linspace=np.linspace
fft=np.fft.fft
ifft=np.fft.ifft
floor=np.floor
rndint=np.random.randint
arr=np.array
ones=np.ones
sinc=np.sinc
arcsinh=np.arcsinh
kron=np.kron
matmul=np.matmul
roll=np.roll
arange=np.arange
resample=signal.resample
resample_poly=signal.resample_poly
upfirdn=signal.upfirdn
rrcosfilter=cp.filters.rrcosfilter
rcosfilter=cp.filters.rcosfilter
ceil=np.ceil
#%%
'''
Beginning of CFM
'''
'''
[
 [Span1_Ch1,Span1_Ch2],
 [Span2_Ch1,Span2_Ch2]
]

e.g. G_bar[2][3] refers to *G_bar* at the 3rd span and the 4th channel

The upper index anywhere, denotes the number of rows or the length of the lists.
The lower index anywhere, denotes the number of columns or the length of the inner lists.

e.g. for f_comb:
    f_comb=[[1,3,5],[3,4,7,8]]
    
    The 1st span contains lambdas [1,3,5] and the 2nd span contains lambdas [3,4,7,8]
'''
#%%
'''
Beginning of SSFM
'''
class Transceiver:
    
#    def __init__(self,TransceiverID,Wavelength,SymbolRate,ChannelBandwidth,isCUT=False):
    def __init__(self,TransceiverID,Wavelength,SymbolRate,ChannelBandwidth):
        self.ID=TransceiverID
        self.Wavelength=Wavelength # Transceiver channel center frequency
        self.SymbolRate=SymbolRate
        self.ChannelBandwidth=ChannelBandwidth # Transceiver channel bandwidth
        self.TxModule=self.Tx()
        self.RxModule=self.Rx()
        self.upsampling_factor=2 # fsampling is twice as much as symbol rate to cover (1+roll_off)*symbolrate
        self.RRCSamplingFrequency=self.upsampling_factor*SymbolRate
#        self.isCUT_Transceiver=isCUT # Determining the CUT transceiver
#        self.isActive=isActive
        self.RRC_trunclength=0 # Must be EVEN!
        
        self.lenTransSignal=0
        
        self.TxModule.Wavelength=Wavelength
        self.RxModule.Wavelength=Wavelength
        
        self.TxModule.SymbolRate=SymbolRate
        self.RxModule.SymbolRate=SymbolRate
        
        self.TxModule.ChannelBandwidth=ChannelBandwidth
        self.RxModule.ChannelBandwidth=ChannelBandwidth
        
    def setTxParams(self,Power_dBm,n_sym,roll_off_factor,mod_type='QPSK',ModulationSymbols=[]):
        self.TxModule.mod_type=mod_type
        self.TxModule.n_sym=n_sym
        self.TxModule.Power_dBm=Power_dBm
        self.TxModule.roll_off_factor=roll_off_factor
        self.TxModule.ModulationSymbols=ModulationSymbols
        
        self.RxModule.roll_off_factor=roll_off_factor
        
        # Temporary
        if roll_off_factor<0.1:
#            self.RRC_trunclength=800
            self.RRC_trunclength=2000
        else:
            self.RRC_trunclength=100
        
        self.RxModule.Power_dBm=Power_dBm
        
        self.lenTransSignal=self.RRC_trunclength+n_sym*self.upsampling_factor
#        self.isremovedTransient=isremovedTransient
        
    def setRxParams(self,roll_off_factor):
        self.RxModule.roll_off_factor=roll_off_factor
        self.TxModule.roll_off_factor=roll_off_factor
#        self.TxModule.mod_type=mod_type
#        self.TxModule.n_sym=n_sym
#        self.TxModule.Power_dBm=Power_dBm
#        self.TxModule.roll_off_factor=roll_off_factor
#        self.TxModule.ModulationSymbols=ModulationSymbols
    
    def TransSignal(self):
            
        Power=10**(self.TxModule.Power_dBm*0.1-3)
        
#        print(Power)
        '''
        Power is considered as dual polarized, so that eahc polarization share
        is half the power amount. The division is by 4.
        '''

        if list(self.TxModule.ModulationSymbols)==[]:
        
            if self.TxModule.mod_type=='16QAM':
                self.TxModule.ModulationSymbols=2*rndint(0,4,[2,self.TxModule.n_sym,2])-3
                self.TxModule.ModulationSymbols=matmul(self.TxModule.ModulationSymbols*sqrt(Power/20),[1,1j])
            
            elif self.TxModule.mod_type=='QPSK' or self.TxModule.mod_type=='4QAM':
#                self.TxModule.ModulationSymbols=2*rndint(0,2,[2,self.TxModule.n_sym,2])-1
                self.TxModule.ModulationSymbols=2*rndint(0,2,[2,self.TxModule.n_sym])-1+1j*(2*rndint(0,2,[2,self.TxModule.n_sym])-1)
#                self.TxModule.ModulationSymbols=matmul(self.TxModule.ModulationSymbols*sqrt(Power/4),[1,1j])
                self.TxModule.ModulationSymbols*=sqrt(Power/4)
            
            else:
                assert 1==2, 'Unknown Modulation Format'
        
#        k=20
        
#        Ts=1/self.SymbolRate # Symbol Period
#        self.RRCSamplingFrequency=self.upsampling_factor*(1+self.TxModule.roll_off_factor)*self.SymbolRate
        
        '''
        Filter Design
        '''
#        HalfNumDeletedSymbols_PreExp=0
        
        
#        ModulationSymbols_Resampled=kron(self.TxModule.ModulationSymbols,[1]+[0]*(self.upsampling_factor-1))
        ModulationSymbols_Resampled=kron(self.TxModule.ModulationSymbols,[1,0])
        
#        print(self.RRC_trunclength)
        if self.TxModule.roll_off_factor==0:
            rrc_filter=sinc(arange(-self.RRC_trunclength/2,self.RRC_trunclength/2,1)/2)
        else:
            t,rrc_filter=rrcosfilter(self.RRC_trunclength+2,self.TxModule.roll_off_factor,1/self.SymbolRate,self.RRCSamplingFrequency)
        
        rrc_filter=rrc_filter[1:]
        
#        rrc_filter=[0]*20+[1]+[0]*19
#        print(rrc_filter)
        
        baseband=signal.upfirdn(rrc_filter,ModulationSymbols_Resampled)
        
#        print(rrc_filter)
        
        self.TxModule.rrc_filter=rrc_filter
        self.RxModule.rrc_filter=rrc_filter
        
        return baseband
        
    class Tx:
        
        def __init__(self):
            '''Primary Parameters'''
            self.Wavelength=0
            self.SymbolRate=0
            self.ChannelBandwidth=0
            '''Secondary Parameters'''
            self.n_sym=1
            self.Power_dBm=0
            self.mod_type='QPSK'
            self.ModulationSymbols=[]
            self.roll_off_factor=0
            self.rrc_filter=0
            
    class Rx:
        
        def __init__(self):
            '''Primary Parameters'''
            self.Wavelength=0
            self.SymbolRate=0
            self.ChannelBandwidth=0
            '''Secondary Parameters'''
            self.rrc_filter=0
            self.roll_off_factor=0
            self.Power_dBm=0
            
        def to_baseband(self,ReceivedSignal,time_D):
#            len_signal=len(ReceivedSignal[0])
#            t=arange(len_signal)/fsampling
#            print(self.Wavelength)
            temp_to_baseband=ReceivedSignal*exp(-2j*pi*self.Wavelength*time_D)
            return temp_to_baseband
        
        def EDC(self,ReceivedSignal,freq_D,beta2,Length):
            return ifft(fft(ReceivedSignal)*exp(-2j*pi**2*freq_D**2*beta2*Length))
        
        def AnalogLowPassFilter(self,ReceivedSignal,fsampling):
            filter_order=91 # Must be ODD!
#            passband_freq=(1-self.roll_off_factor)/2*self.SymbolRate
#            stopband_freq=1.2*self.SymbolRate
#            stopband_freq=0.6*self.SymbolRate
            
            passband_freq=self.SymbolRate*0.6
            stopband_freq=0.7*self.SymbolRate
#            passband_freq=12500000000.0*8/2
#            stopband_freq=125000000000.0/2
            passband_gain=1
            stopband_gain=1e-40
            analogfilter_taps=remez(filter_order, [0, passband_freq, stopband_freq, 0.5*fsampling], [passband_gain, stopband_gain], fs=fsampling)
#            analogfilter_taps=[0]*35+[1]+[0]*34
            w, h = signal.freqz(analogfilter_taps, [1])
#            plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
            output=signal.upfirdn(analogfilter_taps,ReceivedSignal)
            
            # Removing filter tails appended to the signal from it
            if True:
#                HalfFilterTaps=int(filter_order/2-0.5)
                HalfFilterTaps=int(filter_order/2)
                output=array([
                        output[0][HalfFilterTaps:-HalfFilterTaps],
                        output[1][HalfFilterTaps:-HalfFilterTaps]
                        ])
    
#            print('dlfkldsjgklsdjgkldsjgkldsjgklsdjglksjglkg',len(ReceivedSignal[0]))
#            print('dlfkldsjgklsdjgkldsjgkldsjgklsdjglksjglkg',len(output[0]))
            
            return output,filter_order,0.5*fsampling*w/np.pi,20*np.log10(np.abs(h)),h,analogfilter_taps

        
        
        def Downsample(self,ReceivedSignal,fsampling):
            
#            print(self.SymbolRate)
#            numSamplesdn=int(len(ReceivedSignal[0])*self.SymbolRate/fsampling)
#            self.
#            t,rrc_filter=cp.filters.rrcosfilter(2*20,self.roll_off_factor,1/self.SymbolRate,self.SymbolRate)
#            ReceivedSignal=resample(ReceivedSignal.T,numSamplesdn).T
            dnsampling_ratio=int(fsampling/2/self.SymbolRate)
#            print(dnsampling_ratio)
#            print('asasasadgfkdsjgkf',dnsampling_ratio)
#            print('probe1=',sum(np.var(ReceivedSignal,1)))
            output=resample_poly(ReceivedSignal.T,up=1,down=dnsampling_ratio).T
            #%%
            '''
            Right before RRC
            '''
#            print('probe2=',sum(np.var(output,1)))
#            print(len(output[0]))
#            if isMatchedFilterinTx==False:
            #%% RRC Matched filter output
            output=upfirdn(self.rrc_filter,output)
            '''
            Right after RRC
            '''
    #            print('probe3=',sum(np.var(output,1))/sum(self.rrc_filter**2))
            RRCFilterLength=len(self.rrc_filter)
#            HalfRRCFilterLength=int(len(self.rrc_filter)/2)
            
            output=array([
                    output[0][RRCFilterLength-1:-RRCFilterLength],
                    output[1][RRCFilterLength-1:-RRCFilterLength]
                    ])
#            output=array([
#                    output[0][HalfRRCFilterLength:1-HalfRRCFilterLength],
#                    output[1][HalfRRCFilterLength:1-HalfRRCFilterLength]
#                    ])
            
            return output#upfirdn(rrc_filter,yyy,down=int(fsampling/self.SymbolRate),up=1)
        
        def SymbolDetector(self,ReceivedSignal,TotalNumOfSymbols,NumDeletedSyms):
            '''
            Extracting received symbols from the received signal
            '''
#            HalfRRCFilterLength=int(len(self.rrc_filter)/2)
            
#            Detected_ModulationSymbols=array([
#                    ReceivedSignal[0][HalfRRCFilterLength:HalfRRCFilterLength+(TotalNumOfSymbols-NumDeletedSyms-1)*2+1:2],
#                    ReceivedSignal[1][HalfRRCFilterLength:HalfRRCFilterLength+(TotalNumOfSymbols-NumDeletedSyms-1)*2+1:2]
#                    ])/sum(self.rrc_filter**2)
            
            # Regularly picking symbols alternatively 
            Detected_ModulationSymbols=array([
                    ReceivedSignal[0][:(TotalNumOfSymbols-NumDeletedSyms-1)*2+1:2],
                    ReceivedSignal[1][:(TotalNumOfSymbols-NumDeletedSyms-1)*2+1:2]
                    ])#/sum(self.rrc_filter**2)
            
            return Detected_ModulationSymbols
        
        def DecisionMaker(self,DetectedModulationSymbols):
            '''
            This function is not used in the main code, however it may come
            handy later....
            '''
            power_linear=10**(0.1*self.Power_dBm-3)
            
            rx_real=real(DetectedModulationSymbols)
            rx_imag=imag(DetectedModulationSymbols)
            
            det_mod_syms_real=(rx_real>0)*sqrt(power_linear/4)+(rx_real<0)*(-sqrt(power_linear/4))
            det_mod_syms_imag=(rx_imag>0)*sqrt(power_linear/4)+(rx_imag<0)*(-sqrt(power_linear/4))
            
            det_mod_syms=det_mod_syms_real+1j*det_mod_syms_imag
            
            return det_mod_syms
#        def SymbolDetector(self,ReceivedSignal,TotalNumOfSymbols,NumDeletedSyms):
        def PhaseRotationCompensator(self,RxModulationSymbols,TxModulationSymbols):
            '''
            The optimum phase rotation angle is that calculated through xy* where
            "x" denotes transmitted and "y" denoted received.
            '''
#            print('aksalkslaksl=',DetectedModulationSymbols)
#            print('aksalkslaksljskajskajs=',ModulationSymbols)
            opt_angle=np.array([
                    np.angle(np.sum(TxModulationSymbols*conj(RxModulationSymbols),1))
                    ]).T
            return RxModulationSymbols*np.exp(1j*opt_angle),opt_angle
#%%
class Node:
    
    def __init__(self,NodeID):
        self.ID=NodeID
        self.WavelengthDict={}
        self.SymbolRateDict={}
        self.ChannelBandwidthDict={}
        self.TransceiverDict={}
        self.isActiveTransceiverDict={}
#        self.isCUTList=[] # Not really important, it is only used to be summed over its elements
        self.NumofTransceivers=0
        self.fsampling=0
        
    def addNewTransceiver(self,TransceiverID,Wavelength,SymbolRate,ChannelBandwidth):
        self.TransceiverDict[TransceiverID]=Transceiver(TransceiverID,Wavelength,SymbolRate,ChannelBandwidth)
#        self.isCUTList.append(isCUT)
#        self.WavelengthDict[TransceiverID]=Wavelength
#        self.SymbolRateDict[TransceiverID]=SymbolRate
#        self.ChannelBandwidthDict[TransceiverID]=ChannelBandwidth
#        self.isActiveTransceiverDict[TransceiverID]=isActive
        self.isActiveTransceiverDict[TransceiverID]=False
        self.NumofTransceivers+=1
    
#    def toggleTransceiver_on_off_CUT(self,TransceiverID,isActive=True,isCUT=False):
    def toggleTransceiver_on_off(self,TransceiverID,isActive=True):
        
#        if isCUT==True:
#            isActive=True
        
        self.isActiveTransceiverDict[TransceiverID]=isActive
        if isActive==True:
            self.WavelengthDict[TransceiverID]=self.TransceiverDict[TransceiverID].Wavelength
            self.SymbolRateDict[TransceiverID]=self.TransceiverDict[TransceiverID].SymbolRate
            self.ChannelBandwidthDict[TransceiverID]=self.TransceiverDict[TransceiverID].ChannelBandwidth
        
        else:
            try:
                self.WavelengthDict.pop(TransceiverID)
            except:
                pass
            try:
                self.SymbolRateDict.pop(TransceiverID)
            except:
                pass
            try:
                self.ChannelBandwidthDict.pop(TransceiverID)
            except:
                pass
                
#        self.isremovedTransient=isremovedTransient
        
#    def setTransceiverTx(self,TransceiverID,Power_dBm,n_sym,roll_off_factor,mod_type='QPSK',ModulationSymbols=[]):
#        self.TransceiverList[TransceiverID].setTxParams(Power_dBm,n_sym,roll_off_factor,mod_type,ModulationSymbols)
#        
#    def setTransceiverRx(self):
#        return
    
    def Total_addedSignal(self,NumDeletedSyms=0):
        
#        if sum(self.isCUTList)>1:
#            assert 1==2, 'More than one CUT transceiver defined at node '+str(self.ID)
        
        MAX_lambda=max(self.WavelengthDict.values())
        MIN_lambda=min(self.WavelengthDict.values())
        
        ChBW_of_MAX_lambda=self.ChannelBandwidthDict[max(self.WavelengthDict, key=self.WavelengthDict.get)]
        ChBW_of_MIN_lambda=self.ChannelBandwidthDict[min(self.WavelengthDict, key=self.WavelengthDict.get)]
        
#        coeff=1
#        fsampling_temp=coeff*(MAX_lambda-MIN_lambda+ChBW_of_MAX_lambda/2+ChBW_of_MIN_lambda/2)
        
        # Sampling frequency
        fsampling_temp=MAX_lambda-MIN_lambda+ChBW_of_MAX_lambda/2+ChBW_of_MIN_lambda/2
        
#        print(self.fsampling)
        
#        print(self.fsampling/R0/10)
        
#        NumDeletedSyms=0
        for transceiver in self.TransceiverDict.values():
            
#            transceiver.TxModule.ModulationSymbols=transceiver.TxModule.ModulationSymbols[NumDeletedSyms:]
            
#            if transceiver.isCUT_Transceiver==True:
            if 1:
                
                '''
                This is the same int(fsampling_temp/transceiver.RRCSamplingFrequency)
                '''
                RatioOfSamPerSymConversion=ceil(fsampling_temp/transceiver.RRCSamplingFrequency)
                
#                print(RatioOfSamPerSymConversion)
#                print(PreChannelDeletionSamples)
                
                fsampling_temp=RatioOfSamPerSymConversion*transceiver.RRCSamplingFrequency
                
                # Length of signal sent by CUT (a reference for other concurrent signals
                # to fit their lengths with)
#                lenCUTSignal=int(len(transceiver.TransSignal()[0])*RatioOfSamPerSymConversion)
                lenCUTSignal=int(transceiver.lenTransSignal*RatioOfSamPerSymConversion)
                
#                print(lenCUTSignal)
                
#                f_CUT=transceiver.Wavelength
                
#                break
            
        # Building discrete frequency used in both NLSE solver and EDC
        
        f_min=MIN_lambda-ChBW_of_MIN_lambda/2
        freq_D1=np.arange(lenCUTSignal)/lenCUTSignal-f_min/fsampling_temp
#        freq_D1=np.arange(lenCUTSignal)/lenCUTSignal+0*np.ceil(f_min/fsampling_temp)-f_min/fsampling_temp

        freq_D=(freq_D1-np.floor(freq_D1))*fsampling_temp+f_min
        
        time_D=arange(lenCUTSignal)/fsampling_temp
        
        temp_Total_addedSignal=0
        
        for transceiver in self.TransceiverDict.values():
            
            temp_transceiverSignal=transceiver.TransSignal()
            
            dnsampling_ratio=1000
            upsampling_ratio=int(dnsampling_ratio*fsampling_temp/transceiver.RRCSamplingFrequency)
            
            
            
#            print(upsampling_ratio)
            
            temp_transceiverSignal_resampled=resample_poly(temp_transceiverSignal.T,up=upsampling_ratio,down=dnsampling_ratio).T
    
            '''
            Symbol removal only from the beginning
            '''
            temp_transceiverSignal_resampled=array([
                    temp_transceiverSignal_resampled[0][len(temp_transceiverSignal_resampled[0])-lenCUTSignal:],
                    temp_transceiverSignal_resampled[1][len(temp_transceiverSignal_resampled[0])-lenCUTSignal:]
                    ])
            
#            tempo.append(temp_transceiverSignal_resampled)
            
            # Setting the CUT at baseband while rearranging the other channels in the 
            # left and right sides of it
#            temp_transceiverSignal_resampled=temp_transceiverSignal_resampled*exp(2j*pi*(transceiver.Wavelength-f_CUT)*t)
            temp_transceiverSignal_resampled=temp_transceiverSignal_resampled*exp(2j*pi*transceiver.Wavelength*time_D)
            temp_Total_addedSignal=temp_Total_addedSignal+temp_transceiverSignal_resampled
            
#            ccc=1
            
#            print(len(temp_Total_addedSignal[0]))
            self.fsampling=fsampling_temp
            
            
        return temp_Total_addedSignal,freq_D,time_D#1,freq_D2,freq_D3,freq_D4#,tempo#,tempo,self.fsampling
#%%
class Link:
    
    '''
    Unidirectional Link
    
    Link ID = (InNode,OutNode)
    '''
    
    def __init__(self,InNode=None,OutNode=None):
        self.LinkID=(InNode,OutNode)
        self.SpanIDList=[]
        self.SpanList=[]
        self.alphaList=[]
        self.beta2List=[]
        self.gammaList=[]
        self.LengthList=[]
        self.AmplifierGainList=[]
        self.AmplifierNoiseFigureList=[]
        self.NumSpan=0
        self.TupleDict={}
        self.LaunchPowerDict={}
        self.LightPathBandwidthDict={}
        
    def addSpan(self,SpanID,Length,alpha,beta2,gamma,AmplifierNoiseFigure=-1000,AmplifierGain_dB=''):
        self.SpanIDList.append(SpanID)
        self.SpanList.append(self.Span(SpanID,Length,alpha,beta2,gamma,AmplifierNoiseFigure,AmplifierGain_dB))
        self.alphaList.append(alpha)
        self.beta2List.append(beta2)
        self.gammaList.append(gamma)
        self.LengthList.append(Length)
        self.AmplifierGainList.append(AmplifierGain_dB)
        self.AmplifierNoiseFigureList.append(AmplifierNoiseFigure)
        self.NumSpan+=1
        
    class Span:
    
        def __init__(self,SpanID,Length,alpha,beta2,gamma,AmplifierNoiseFigure=-1000,AmplifierGain_dB=''):
            self.SpanID=SpanID
            self.alpha=alpha
            self.beta2=beta2
            self.gamma=gamma
            self.Length=Length
            self.AmplifierNoiseFigure=AmplifierNoiseFigure
            self.AmplifierGain_dB=AmplifierGain_dB
#%%
class LightPath:
    
    def __init__(self):
        self.Wavelength=None
        self.LPNodeList=[]
        self.LaunchPower=None
        self.LightPathBandwidth=None
        self.ModulationType=None
        self.LPLinkList=[]
#%%
class Network:
    
    def __init__(self):
        self.NodeDict={}
        self.LinkDict={}
        self.LightPathDict={}
        self.NumofNodes=0
        self.NumofLinks=0
    
    def addNode(self,NodeID):
        self.NodeDict[NodeID]=Node(NodeID)
        self.NumofNodes+=1
        
    def addLink(self,InNode,OutNode):
        self.LinkDict[(InNode,OutNode)]=Link(InNode,OutNode)
        self.NumofLinks+=1
        
    def addLightPath(self,LPID,LPNodeList,TransceiverID,LaunchPower):
        # Initializing lightpath
        self.LightPathDict[LPID]=LightPath()
        # Lightpath source node
        SourceNodeID=LPNodeList[0]
        # Updating the lightpath ingredient
        self.LightPathDict[LPID].Wavelength=self.NodeDict[SourceNodeID].TransceiverDict[TransceiverID].Wavelength
        self.LightPathDict[LPID].LightPathBandwidth=self.NodeDict[SourceNodeID].TransceiverDict[TransceiverID].ChannelBandwidth
        self.LightPathDict[LPID].ModulationType=self.NodeDict[SourceNodeID].TransceiverDict[TransceiverID].TxModule.mod_type
        # Toggling on the corresponding transceiver
        self.NodeDict[SourceNodeID].toggleTransceiver_on_off(TransceiverID,isActive=True)
        # Importing nodes in the lightpath
        for iNode in LPNodeList:
            self.LightPathDict[LPID].LPNodeList.append(self.NodeDict[iNode])
        # Importing links in the lightpath
        for iNode,jNode in zip(LPNodeList,LPNodeList[1:]):
            self.LightPathDict[LPID].LPLinkList.append(self.LinkDict[(iNode,jNode)])
            # Updating links in the network
#        for linkind in self.LightPathDict[LPID].LPLinkList:
#            self.LinkDict[linkind].TupleSet.add((Wavelength,LPNodeList[0]))
            self.LinkDict[(iNode,jNode)].TupleDict[LPID]=(self.LightPathDict[LPID].Wavelength,LPNodeList[0])
            self.LinkDict[(iNode,jNode)].LaunchPowerDict[LPID]=LaunchPower
            self.LinkDict[(iNode,jNode)].LightPathBandwidthDict[LPID]=self.LightPathDict[LPID].LightPathBandwidth
        
    def Export(self,LPID=0,BandWidth=1,extype='EGN'):
        
        '''
        To be fixed!!!!!!!!!!
        '''
        if extype=='EGN':
            XLinkDict={}
            XLPA={}
            XLaunchPowerDict={}
            XPhiDict={}
            XPsiDict={}
            for linkid in self.LinkDict:
                tempLink=self.LinkDict[linkid]
                XLinkDict[linkid]=[]
                XLinkDict[linkid].append(tempLink.LengthList)
                XLinkDict[linkid].append(tempLink.alphaList)
                XLinkDict[linkid].append(tempLink.beta2List)
                XLinkDict[linkid].append(tempLink.gammaList)
                XLinkDict[linkid].append(tempLink.AmplifierGainList)
                XLinkDict[linkid].append(tempLink.AmplifierNoiseFigureList)
                XLinkDict[linkid].append(set([(tuple_[0]/BandWidth,tuple_[1]) for tuple_ in tempLink.TupleDict.values()]))
#                XLinkDict[linkid].append(set(list(tempLink.TupleDict.values())))
                XLinkDict[linkid].append(tempLink.NumSpan)
            for LPind in self.LightPathDict:
                tempLP=self.LightPathDict[LPind]
                XLPA[LPind]=[tempLP.Wavelength/BandWidth,tempLP.LPNodeList,list(zip(tempLP.LPNodeList,tempLP.LPNodeList[1:]))]
                XLaunchPowerDict[(tempLP.Wavelength/BandWidth,tempLP.LPNodeList[0])]=tempLP.LaunchPower
                XPhiDict[(tempLP.Wavelength/BandWidth,tempLP.LPNodeList[0])]=tempLP.Phi
                XPsiDict[(tempLP.Wavelength/BandWidth,tempLP.LPNodeList[0])]=tempLP.Psi
            # These outputs are used for EGN simulation
            return XLinkDict,XLPA,XLaunchPowerDict,XPhiDict,XPsiDict
        elif extype=='CFM':
            '''
            *LinkDict* in this return type contains lists of link specs along a
            specified lightpath
            '''
            tempLPA=self.LightPathDict[LPID]
            XSpansSpec={}
            XSpansSpec['Length']=[]
            XSpansSpec['alpha']=[]
            XSpansSpec['beta2']=[]
            XSpansSpec['gamma']=[]
            XSpansSpec['AmplifierGain']=[]
            XSpansSpec['AmplifierNoiseFigureList']=[]
            XSpansSpec['f_comb']=[]
            XSpansSpec['R_comb']=[]
            XSpansSpec['Power_comb']=[]
            XSpansSpec['f_CUT']=tempLPA.Wavelength
            for tempLink in tempLPA.LPLinkList:
                XSpansSpec['Length'].extend(tempLink.LengthList)
                XSpansSpec['alpha'].extend(tempLink.alphaList)
                XSpansSpec['beta2'].extend(tempLink.beta2List)
                XSpansSpec['gamma'].extend(tempLink.gammaList)
                XSpansSpec['AmplifierGain'].extend(tempLink.AmplifierGainList)
                XSpansSpec['AmplifierNoiseFigureList'].extend(tempLink.AmplifierNoiseFigureList)
                XSpansSpec['f_comb'].extend([array(list(tempLink.TupleDict.values()))[:,0]]*tempLink.NumSpan)
                XSpansSpec['R_comb'].extend([list(tempLink.LightPathBandwidthDict.values())]*tempLink.NumSpan)
                XSpansSpec['Power_comb'].extend([list(tempLink.LaunchPowerDict.values())]*tempLink.NumSpan)
            return XSpansSpec
#%%
'''
Your code writing starts from here!
'''
def NLSE_Solver_Link(inputSignal,freq_D,alpha,beta2,gamma,lspan,numstep,nspan,NF,fsampling):
    
    
    
    '''
    By writing the SSFM solver core, implement proper operations (
    linear impairments, non-linear impairments and ASE noise addition) 
    on an "inputSignal" to generate an "outputSignal"
    '''
    amp_gain=np.exp(alpha*lspan)
    
    if beta2==0 or gamma==0:
        numstep=1
    
    len_step=lspan/numstep
    
    if alpha==0:
        len_step_eff=len_step
    else:
        len_step_eff=(1-np.exp(-alpha*len_step))/alpha
    
    half_exp=np.exp(-alpha*len_step/4+1j*np.pi**2*beta2*len_step*freq_D**2)
    full_exp=np.exp(-alpha*len_step/2+2j*np.pi**2*beta2*len_step*freq_D**2)
    
    propagated_signal=inputSignal.copy()
    progressbar=tqdm(total=nspan*numstep,leave=True,position=0)
    for i in range(nspan):
        propagated_signal=ifft(fft(propagated_signal)*half_exp)
        for j in range(numstep):
            propagated_signal*=np.exp(1j*8/9*gamma*len_step_eff*sum(abs(propagated_signal)**2))
            propagated_signal=ifft(fft(propagated_signal)*full_exp)
            progressbar.update(1)
        propagated_signal=ifft(fft(propagated_signal)/half_exp)
        propagated_signal*=amp_gain**0.5
        
        h_planck=6.626e-34
        nu_ase=3e8/1550e-9
        ase_var=h_planck*nu_ase*amp_gain*10**(0.1*NF)*fsampling
        ase_noise=[[1,1j,0,0],[0,0,1,1j]]@np.random.randn(4,len(propagated_signal[0]))*(ase_var/4)**0.5
        propagated_signal+=ase_noise
    
    return propagated_signal
#%%
if __name__=='__main__':
    plt.close("all")
    
    n_sym=10000
    PowerSet_dBm=arange(10,11,1)
    roll_off_factor=0.02
    ##### Span Parameters #####
    alpha=0.22/4343
    beta2=-21.3e-27*0
    gamma=1.3e-3
    Lspan=100e3
    NF=5.5-1000
    
    ##### Link Parameters ######
    Nspan=25                               # Number of spans per link
    
    ###### Transceiver parameters ############
    SymbolRate=32e9                                  # Symbol rate (Hz)
    ChannelBandwidth=32.7e9    # Channel bandwidth (Hz)
    
    del_margin=200        # Number of transient symbols to be deleted
    SNR_SSFM_dB=[]        # Empty vector to maintain SSFM SNR values
#    SNR_CFM_dB=[]         # Empty vector to maintain CFM SNR values
    
    numstep=100
    
    '''Defining network'''
    network=Network()
    
    '''Adding nodes to the network'''
    network.addNode(1)
    network.addNode(2)
        
    '''Adding transceivers per nodes'''
    network.NodeDict[1].addNewTransceiver(
            1,
            1*ChannelBandwidth,
            SymbolRate,
            ChannelBandwidth
            )    
    network.NodeDict[2].addNewTransceiver(
            1,
            1*ChannelBandwidth,
            SymbolRate,
            ChannelBandwidth
            )
    
    '''Adding links to the network'''
    network.addLink(1,2)
    
    '''Adding spans per link'''
    for i in range(Nspan):
        network.LinkDict[(1,2)].addSpan(i,Lspan,alpha,beta2,gamma,NF)
    
    for power_dBm in PowerSet_dBm:
        
        print('....................',power_dBm)
        
        '''Linear power'''
        power_linear                          = 10**(0.1*power_dBm-3)
        
        '''Setting transceivers'''
        network.NodeDict[1].TransceiverDict[1].setTxParams(power_dBm,n_sym,roll_off_factor)
#        network.NodeDict[1].TransceiverDict[1].setTxParams(power_dBm,n_sym,roll_off_factor,ModulationSymbols=tx_syms_dict_from_new_ssfm_at_unit_power["dem1"]*(power_linear**0.5))
        
        '''Adding lightpaths only for turning on transceivers'''
        network.addLightPath(1,[1,2],1,power_dBm)
        
        '''Destination node Rx module'''
        Rx_of_receiver_node                   = network.LightPathDict[1].LPNodeList[1].TransceiverDict[1].RxModule
        
        '''Lightpath accessories for CFM'''
        XSpansSpec                            = network.Export(1,1,'CFM')
        
        '''Tx RRC filter which is also used as matched filter'''
        rrc_filter                            = network.LightPathDict[1].LPNodeList[0].TransceiverDict[1].TxModule.rrc_filter
        
        '''Source node output in point to point transmission'''
        node_output,freq_D,time_D             = network.LightPathDict[1].LPNodeList[0].Total_addedSignal()
        
        '''Setting RRC filter as matched filter in destination node Rx module'''
        Rx_of_receiver_node.rrc_filter        = network.LightPathDict[1].LPNodeList[0].TransceiverDict[1].TxModule.rrc_filter
        
#        node_output=node_output_from_new_ssfm_at_unit_power.copy()*(power_linear**0.5)
        #%%
        '''
        You should use your NLSE solver here to find the fiber output given
        the fiber input.
        
        In this phase, let the input to your NLSE solver function to be
        "node_output" and the output of your NLSE solver function to be
        "link_output"
        
        To apply disperion on signal, you need a discrete frequency vector. Use
        "freq_D" to build the dispersion filter as exp(2j*pi**2*beta2*Length*freq_D**2)
        '''
        link_output=NLSE_Solver_Link(node_output,freq_D,alpha,beta2,gamma,
                                     Lspan,numstep,Nspan,NF,network.NodeDict[1].fsampling)
#         link_output = NLSE_Solver_Link(node_output,freq_D,other_parameters)
        '''
        '''
        #%%
        
#        loi_nu=network.LightPathDict[1].Wavelength
#        len_signal=len(link_output[0])
#        bw=ChannelBandwidth
#        fsampling
#        
#        ##### EDC and LPF #####
#        acc_dispersion=beta2*Lspan*Nspan
#        
#        edc_exp=np.exp(-2j*np.pi**2*acc_dispersion*freq_D**2)
#        edc_output=ifft(fft(link_output)*edc_exp)
#        
#        ##### to-baseband #####
#        to_baseband_output=edc_output*np.exp(-2j*np.pi*loi_nu*time_D)
#        
#        ##### LPF #####
#        N_ones=int(len_signal*bw/2/fsampling)
#        lpf=[1]*N_ones+[0]*(len_signal-2*N_ones)+[1]*N_ones
#        lpf_output=ifft(fft(to_baseband_output)*lpf)
#        
#        ##### MF #####
#        mf_output=np.array([
#                np.convolve(lpf_output[0],rrc_pulse_shape),
#                np.convolve(lpf_output[1],rrc_pulse_shape)
#                ])
#        
#        ##### Rx syms #####
#        rx_syms=np.array([
#                mf_output[0][2*MU:2*MU+(n_sym-1)*2*upsratio+1:2*upsratio],
#                mf_output[1][2*MU:2*MU+(n_sym-1)*2*upsratio+1:2*upsratio]
#                ])
#        tx_syms=network.LightPathDict[1].LPNodeList[0].TransceiverDict[1].TxModule.ModulationSymbols
#        
#        ##### Error #####
#        alpha_x=sum(rx_syms[0]*tx_syms[0].conj())/sum(tx_syms[0]*tx_syms[0].conj())
#        alpha_y=sum(rx_syms[1]*tx_syms[1].conj())/sum(tx_syms[1]*tx_syms[1].conj())
#        
#        noise_x=rx_syms[0]-alpha_x*tx_syms[0]
#        noise_y=rx_syms[1]-alpha_y*tx_syms[1]
#        
#        snr_x=10*np.log10(abs(alpha_x)**2*sum(abs(tx_syms[0])**2)/sum(abs(noise_x)**2))
#        snr_y=10*np.log10(abs(alpha_y)**2*sum(abs(tx_syms[1])**2)/sum(abs(noise_y)**2))
        
#        plt.figure()
#        plt.plot(np.real(rx_syms[0]),np.imag(rx_syms[0]),".")
#        print(snr_x)
        
        '''EDC operation on link output (to remove dispersion)'''
        EDC_output                            = Rx_of_receiver_node.EDC(link_output,freq_D,beta2*Nspan,Lspan)
        
        '''Converting signal to baseband'''
        EDC_output_to_baseband                = Rx_of_receiver_node.to_baseband(EDC_output,time_D)
        
        '''Passing baseband signal through matched filter'''
        EDC_output_to_baseband_Downsampled    = Rx_of_receiver_node.Downsample(EDC_output_to_baseband,network.NodeDict[1].fsampling)
#        dnsampling_ratio=int(fsampling/2/SymbolRate)
#        output1=resample_poly(EDC_output_to_baseband.T,up=1,down=dnsampling_ratio).T
#        output2=upfirdn(rrc_filter_ds,output1)
#        RRCFilterLength=len(rrc_filter_ds)
#        EDC_output_to_baseband_Downsampled=array([
#                output2[0][RRCFilterLength-1:-RRCFilterLength],
#                output2[1][RRCFilterLength-1:-RRCFilterLength]
#                ])
        '''Received modulation symbols at final step'''
        RawRxModSymbols                          = Rx_of_receiver_node.SymbolDetector(EDC_output_to_baseband_Downsampled,n_sym,0)
        
        '''Transmitted modulation symbols from beginning'''
        TxModSymbols                          = network.LightPathDict[1].LPNodeList[0].TransceiverDict[1].TxModule.ModulationSymbols.copy()
        
        '''Compensating phase rotation due to non-linear effect'''
#        RxModSymbols_rotated,opt_angle        = Rx_of_receiver_node.PhaseRotationCompensator(RawRxModSymbols,TxModSymbols)
        
        
        
        
        
        
        
        TxModSymbols=np.array([
                TxModSymbols[0][del_margin:-del_margin],
                TxModSymbols[1][del_margin:-del_margin],
                ])
        RxModSymbols=np.array([
                RawRxModSymbols[0][del_margin:-del_margin],
                RawRxModSymbols[1][del_margin:-del_margin],
                ])
        
        alpha_x=sum(RxModSymbols[0]*TxModSymbols[0].conj())/sum(TxModSymbols[0]*TxModSymbols[0].conj())
        alpha_y=sum(RxModSymbols[1]*TxModSymbols[1].conj())/sum(TxModSymbols[1]*TxModSymbols[1].conj())
        
        noise_x=RxModSymbols[0]-alpha_x*TxModSymbols[0]
        noise_y=RxModSymbols[1]-alpha_y*TxModSymbols[1]
        
        snr_x=10*np.log10(abs(alpha_x)**2*sum(abs(TxModSymbols[0])**2)/sum(abs(noise_x)**2))
        snr_y=10*np.log10(abs(alpha_y)**2*sum(abs(TxModSymbols[1])**2)/sum(abs(noise_y)**2))
        
        SNR_SSFM_dB.append(snr_x)
        
        
        
        
        
        
        
        
        
        
        
        
#        '''Error calculation'''
#        err                                   = RxModSymbols_rotated/2-TxModSymbols
#        
#        
#        
#        '''Removing transient'''
#        err=[
#                err[0][del_margin:-del_margin],
#                err[1][del_margin:-del_margin]
#                ]
#        
#        '''Error variance = Noise variance'''
#        var_err                               = np.sum(np.var(err,1))*2
#        
#        '''Calculating and saving SSFM and CFM SNR values'''
#        SNR_SSFM_dB.append(power_dBm-30-10*np.log10(var_err))
#        SNR_CFM_dB.append(SNR_CFM(XSpansSpec))
    
    #%%
    plt.figure()
    plt.plot(node_output1[0]-node_output[0])
    plt.title("Node output")
    
    plt.figure()
    plt.plot(link_output[0]-propagated_signal[0])
    plt.title("Link output")
    
    plt.figure()
    plt.plot(edc_output[0]-EDC_output[0])
    plt.title("EDC output")
    
    plt.figure()
    plt.plot(lpf_output[0]-EDC_output_to_baseband[0])
    plt.title("to-baseband output")
    
    plt.figure()
    plt.plot(real(RxModSymbols[0]),imag(RxModSymbols[0]),".")
    
    plt.figure()
    plt.plot(SNR_SSFM_dB,"d-",label="SSFM")
    plt.plot(snr_x_list_from_new_ssfm,"o-",label="New SSFM")
    plt.grid("on")
#    plt.plot(SNR_CFM_dB,"o-",label="CFM")
    plt.legend()