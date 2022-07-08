function varargout = opticaltransmissionGUI(varargin)
% OPTICALTRANSMISSIONGUI M-file for opticaltransmissionGUI.fig
%      OPTICALTRANSMISSIONGUI, by itself, creates a new OPTICALTRANSMISSIONGUI or raises the existing
%      singleton*.
%
%      H = OPTICALTRANSMISSIONGUI returns the handle to a new OPTICALTRANSMISSIONGUI or the handle to
%      the existing singleton*.
%
%      OPTICALTRANSMISSIONGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in OPTICALTRANSMISSIONGUI.M with the given input arguments.
%
%      OPTICALTRANSMISSIONGUI('Property','Value',...) creates a new OPTICALTRANSMISSIONGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before opticaltransmissionGUI_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to opticaltransmissionGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help opticaltransmissionGUI

% Last Modified by GUIDE v2.5 25-Apr-2007 12:57:47

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @opticaltransmissionGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @opticaltransmissionGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before opticaltransmissionGUI is made visible.
function opticaltransmissionGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to opticaltransmissionGUI (see VARARGIN)

% Choose default command line output for opticaltransmissionGUI
global Condition
Condition=0;
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes opticaltransmissionGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = opticaltransmissionGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


%%Start/stop button
% --- Executes on button press in startbutton.
function startbutton_Callback(hObject, eventdata, handles)
% hObject    handle to startbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

warning off
global Condition
if Condition==0
    Condition=1;
    set(handles.runindicator,'string','Running...');
    main_function(handles)
else
    Condition=0;
    set(handles.runindicator,'string',' ');
end

function main_function(handles)
global Condition
Q_summation =0;
POWER_summation=0;
counter=0;
while Condition
    back2back=get(handles.back2back,'Value');
    transm=get(handles.transmission,'Value');
    transmdispcomp=get(handles.transmissiondispcomp,'Value');
    if back2back
        image(imread('setupback2back.bmp'),'parent',handles.illustration)
        axis(handles.illustration,'off')
    elseif transm
        image(imread('setupcascampl.bmp'),'parent',handles.illustration)
        axis(handles.illustration,'off')
    else
        image(imread('setupdispcomp.bmp'),'parent',handles.illustration)
        axis(handles.illustration,'off')
    end

    bitrate=str2double(get(handles.bitrate,'string'))*1e9;       %Simulated bit rate (bit/s)
    
    t_d=(round(get(handles.dec_time,'value')*31)+1);
    I_d=get(handles.dec_ampl,'value');
    
    %

% **** Transmission_Main Start **********************************************************************


%*** Input variables *******************
modform=get(handles.modulationformat,'value');
if modform==1
    modformat='NRZ';    %The modulation format to use ('NRZ' or 'RZ')
elseif modform==2
    modformat='RZ';
end
no_bits=str2double(get(handles.nobits,'string'));             %Number of bits to transmit
lambda=str2double(get(handles.wavelength,'string'))*1e-9;     %Optical carrier wavelength (m)
sample_bit=32;                                             %Number of samples per bit 
% c=3e8;                                                     %Speed of light in vacuum (m/s)
L=str2double(get(handles.Lsmf,'string'))*1e3;                 %Fiber length (m)
D=str2double(get(handles.Dsmf,'string'));                     %Fiber dispersion parameter in ps/(km-nm)
S=str2double(get(handles.Ssmf,'string'));                     %Fiber dispersion slope in ps/(km-nm^2)
alpha=str2double(get(handles.attsmf,'string'));               %Fiber attenuation (dB/km)
Ldcf=str2double(get(handles.Ldcf,'string'))*1e3;              %Fiber length (m)
Ddcf=str2double(get(handles.Ddcf,'string'));                  %Fiber dispersion parameter in ps/(km-nm)
Sdcf=str2double(get(handles.Sdcf,'string'));                  %Fiber dispersion slope in ps/(km-nm^2)
alphadcf=str2double(get(handles.attDCF,'string'));            %Fiber attenuation (dB/km)
Tfwhm=str2double(get(handles.fwhm,'string'))*1e-12;           %RZ pulse width (full width half maximum) (s)
Ppeak=10^(str2double(get(handles.power,'string'))/10)*1e-3;   %Optical peak power out from the transmitter (W)
rex=0.0;                                                   %Transmitter extinction ratio
chirp=str2double(get(handles.chirp,'string'));                %Transmitter chirp parameter
Ssp=0;                                                     %Spectral density of noise generated in optical amplifier (initially set to zero) (W/Hz) 
Resp=1;                                                    %Receiver responsitivity (A/W)
optfilter=get(handles.optfilter,'value');
if optfilter==1
    dv_opt=3.75e12;                                        %Bandwidth of amplifier bandwidth (Hz)
elseif optfilter==2
    dv_opt=125e9;                                          %Bandwidth of 1 nm optical filter (Hz)
end
    R=300;                                                 %Receiver load resistance (Ohm)
deltaf=str2double(get(handles.filterBW,'string'))*bitrate;    %Receiver electrical filter bandwidth (Hz)
forder=3;                                                  %Receiver electrical filter order
attenuation=abs(str2double(get(handles.varatt,'string')));        %Variable attenuation (dB)

%*** Input variables (end)*******************


%**** Generate data pulse train to transmit *****************
[pulse_train,time_vector]=Transmitter(modformat,Ppeak,rex,bitrate,no_bits,chirp,Tfwhm);% for RZ pulses: ,chirp,T0

N=length(time_vector);                   %Number of samples
df=1/(time_vector(end)-time_vector(1));  %Frequency resolution
freq_vect=-(N/2)*df:df:(N/2-1)*df;

% *** Back to back case *****
if or(back2back,str2double(get(handles.spans,'string'))==0)
    pulse_train=pulse_train*10^(-attenuation/20); %Attenuating pulse train
    POWER=10*log10(mean(abs(pulse_train).^2*1e3,2));
    set(handles.optpower,'string',num2str(round(POWER*10)/10))
    if get(handles.preamplifier,'value')  %
        [pulse_train,Ssp,~]=EDFA(pulse_train,Ssp,lambda);
    end
    
% *** Transmission w/o dispersion compensation    
elseif transm
    set(handles.preamplifier,'value',1) 
    for q=1:str2double(get(handles.spans,'string'))
        [pulse_train,Ssp]=FiberPropagation2(pulse_train,Ssp,alpha,D,S,L,time_vector,lambda);
        if q==str2double(get(handles.spans,'string'))  %Received opt. power after transmission in SMF
            POWER=10*log10(mean(abs(pulse_train).^2*1e3,2));
            set(handles.optpower,'string',num2str(round(POWER*10)/10))
        end
        [pulse_train,Ssp,~]=EDFA(pulse_train,Ssp,lambda);
    end
    
% *** Transmission w/ dispersion compensation     
else
   set(handles.preamplifier,'value',1)
   for q=1:str2double(get(handles.spans,'string'))
        [pulse_train,Ssp]=FiberPropagation2(pulse_train,Ssp,alpha,D,S,L,time_vector,lambda);
        if q==str2double(get(handles.spans,'string')) %Received opt. power after transmission in SMF
            POWER=10*log10(mean(abs(pulse_train).^2*1e3,2));
            set(handles.optpower,'string',num2str(round(POWER*10)/10))
        end
        [pulse_train,Ssp,~]=EDFA(pulse_train,Ssp,lambda);
        [pulse_train,Ssp]=FiberPropagation2(pulse_train,Ssp,alphadcf,Ddcf,Sdcf,Ldcf,time_vector,lambda);
        [pulse_train,Ssp,~]=EDFA(pulse_train,Ssp,lambda);
   end
end
    


%**** Receive transmitted pulse train ***********************
[Iout,Is]=Receiver(pulse_train,Resp,time_vector,dv_opt,R,Ssp,forder,deltaf);

% Plot signal spectrum
if get(handles.specwithfilter,'value')
    if get(handles.specwithoutfilter,'value')
        semilogy(handles.spectrum,freq_vect*1e-9,fftshift(abs(fft(Is))),'r',freq_vect*1e-9,fftshift(abs(fft(Iout))),'b')
        xlabel(handles.spectrum,'Frequency (GHz)')
    else
       semilogy(handles.spectrum,freq_vect*1e-9,fftshift(abs(fft(Iout))),'b')
       xlabel(handles.spectrum,'Frequency (GHz)')
    end
elseif get(handles.specwithoutfilter,'value')
    semilogy(handles.spectrum,freq_vect*1e-9,fftshift(abs(fft(Is))),'r')
    xlabel(handles.spectrum,'Frequency (GHz)')
end
set(handles.spectrum,'Xlim',[-4*bitrate*1e-9 4*bitrate*1e-9])

eyematrix=reshape(Iout,sample_bit*2,no_bits/2);  %Creates a matrix with two bits in each row
t=1:32*no_bits;
plot(handles.eyediagram,t(1:2*sample_bit)*1/bitrate/32*1e12,eyematrix,'-b')  %Plots eye-diagram

xlabel(handles.eyediagram,'Time (ps)')
ylim=get(handles.eyediagram,'Ylim');
set(handles.eyediagram,'Xlim',[0 2*1/bitrate*1e12])
hold(handles.eyediagram,'on');

plot(handles.eyediagram,[t_d-0.5 t_d-0.5]/32*1/bitrate*1e12,ylim,'r:',[t_d+0.5 t_d+0.5]/32*1/bitrate*1e12,ylim,'r:',[t_d t_d+32]/32*1/bitrate*1e12,((ylim(2)-ylim(1))*I_d+ylim(1))*[1 1],'rx','markersize',10);hold(handles.eyediagram,'off'); %Plots current decision point

[Q,Qdata,~]=Qvalue(Iout,no_bits,t_d,(ylim(2)-ylim(1))*I_d+ylim(1));

[h1,b1]=hist(Qdata,50);
barh(handles.histogram,b1,h1);
set(handles.histogram,'Ylim',ylim)
set(handles.Qvalue,'String',num2str(round(100*Q)/100))  %Plots the histogram of the received data at decision time t_d

% ******* Transmission_Main end ********




if Condition
    Q_summation=Q_summation+Q;
    POWER_summation=POWER_summation+POWER;
    counter=counter+1;
else
    POWER_summation=0;
    Q_summation=0;
    counter=0;
end
disp(['Q = ' num2str(Q_summation/counter) ...
    '       Rx. Power = ' num2str(POWER_summation/counter)]);
% disp(counter)
pause(0.1)
end



% --- Executes on slider movement.
function dec_time_Callback(hObject, eventdata, handles)
% hObject    handle to dec_time (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function dec_time_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dec_time (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function dec_ampl_Callback(hObject, eventdata, handles)
% hObject    handle to dec_ampl (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function dec_ampl_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dec_ampl (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function Ddcf_Callback(hObject, eventdata, handles)
% hObject    handle to Ddcf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Ddcf as text
%        str2double(get(hObject,'String')) returns contents of Ddcf as a double


% --- Executes during object creation, after setting all properties.
function Ddcf_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Ddcf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end











function Qvalue_Callback(hObject, eventdata, handles)
% hObject    handle to Qvalue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Qvalue as text
%        str2double(get(hObject,'String')) returns contents of Qvalue as a double


% --- Executes during object creation, after setting all properties.
function Qvalue_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Qvalue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1


% --- Executes on button press in radiobutton2.
function radiobutton2_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton2



function modulationformat_Callback(hObject, eventdata, handles)
% hObject    handle to modulationformat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of modulationformat as text
%        str2double(get(hObject,'String')) returns contents of modulationformat as a double


% --- Executes during object creation, after setting all properties.
function modulationformat_CreateFcn(hObject, eventdata, handles)
% hObject    handle to modulationformat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function nobits_Callback(hObject, eventdata, handles)
% hObject    handle to nobits (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nobits as text
%        str2double(get(hObject,'String')) returns contents of nobits as a double


% --- Executes during object creation, after setting all properties.
function nobits_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nobits (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function bitrate_Callback(hObject, eventdata, handles)
% hObject    handle to bitrate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bitrate as text
%        str2double(get(hObject,'String')) returns contents of bitrate as a double


% --- Executes during object creation, after setting all properties.
function bitrate_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bitrate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function power_Callback(hObject, eventdata, handles)
% hObject    handle to power (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of power as text
%        str2double(get(hObject,'String')) returns contents of power as a double


% --- Executes during object creation, after setting all properties.
function power_CreateFcn(hObject, eventdata, handles)
% hObject    handle to power (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function chirp_Callback(hObject, eventdata, handles)
% hObject    handle to chirp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of chirp as text
%        str2double(get(hObject,'String')) returns contents of chirp as a double


% --- Executes during object creation, after setting all properties.
function chirp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to chirp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function fwhm_Callback(hObject, eventdata, handles)
% hObject    handle to fwhm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of fwhm as text
%        str2double(get(hObject,'String')) returns contents of fwhm as a double


% --- Executes during object creation, after setting all properties.
function fwhm_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fwhm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function wavelength_Callback(hObject, eventdata, handles)
% hObject    handle to wavelength (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of wavelength as text
%        str2double(get(hObject,'String')) returns contents of wavelength as a double


% --- Executes during object creation, after setting all properties.
function wavelength_CreateFcn(hObject, eventdata, handles)
% hObject    handle to wavelength (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function attsmf_Callback(hObject, eventdata, handles)
% hObject    handle to attsmf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of attsmf as text
%        str2double(get(hObject,'String')) returns contents of attsmf as a double


% --- Executes during object creation, after setting all properties.
function attsmf_CreateFcn(hObject, eventdata, handles)
% hObject    handle to attsmf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function filterBW_Callback(hObject, eventdata, handles)
% hObject    handle to filterBW (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of filterBW as text
%        str2double(get(hObject,'String')) returns contents of filterBW as a double


% --- Executes during object creation, after setting all properties.
function filterBW_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filterBW (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in optfilter.
function optfilter_Callback(hObject, eventdata, handles)
% hObject    handle to optfilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns optfilter contents as cell array
%        contents{get(hObject,'Value')} returns selected item from optfilter


% --- Executes during object creation, after setting all properties.
function optfilter_CreateFcn(hObject, eventdata, handles)
% hObject    handle to optfilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function spans_Callback(hObject, eventdata, handles)
% hObject    handle to spans (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of spans as text
%        str2double(get(hObject,'String')) returns contents of spans as a double


% --- Executes during object creation, after setting all properties.
function spans_CreateFcn(hObject, eventdata, handles)
% hObject    handle to spans (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Ldcf_Callback(hObject, eventdata, handles)
% hObject    handle to Ldcf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Ldcf as text
%        str2double(get(hObject,'String')) returns contents of Ldcf as a double


% --- Executes during object creation, after setting all properties.
function Ldcf_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Ldcf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Lsmf_Callback(hObject, eventdata, handles)
% hObject    handle to Lsmf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Lsmf as text
%        str2double(get(hObject,'String')) returns contents of Lsmf as a double


% --- Executes during object creation, after setting all properties.
function Lsmf_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Lsmf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





function Dsmf_Callback(hObject, eventdata, handles)
% hObject    handle to Dsmf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Dsmf as text
%        str2double(get(hObject,'String')) returns contents of Dsmf as a double


% --- Executes during object creation, after setting all properties.
function Dsmf_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Dsmf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function attDCF_Callback(hObject, eventdata, handles)
% hObject    handle to attDCF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of attDCF as text
%        str2double(get(hObject,'String')) returns contents of attDCF as a double


% --- Executes during object creation, after setting all properties.
function attDCF_CreateFcn(hObject, eventdata, handles)
% hObject    handle to attDCF (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Ssmf_Callback(hObject, eventdata, handles)
% hObject    handle to Ssmf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Ssmf as text
%        str2double(get(hObject,'String')) returns contents of Ssmf as a double


% --- Executes during object creation, after setting all properties.
function Ssmf_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Ssmf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Sdcf_Callback(hObject, eventdata, handles)
% hObject    handle to Sdcf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Sdcf as text
%        str2double(get(hObject,'String')) returns contents of Sdcf as a double


% --- Executes during object creation, after setting all properties.
function Sdcf_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Sdcf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





function varatt_Callback(hObject, eventdata, handles)
% hObject    handle to varatt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of varatt as text
%        str2double(get(hObject,'String')) returns contents of varatt as a double


% --- Executes during object creation, after setting all properties.
function varatt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to varatt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in specwithoutfilter.
function specwithoutfilter_Callback(hObject, eventdata, handles)
% hObject    handle to specwithoutfilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of specwithoutfilter


% --- Executes on button press in specwithfilter.
function specwithfilter_Callback(hObject, eventdata, handles)
% hObject    handle to specwithfilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of specwithfilter


% --- Executes on button press in preamplifier.
function preamplifier_Callback(hObject, eventdata, handles)
% hObject    handle to preamplifier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of preamplifier





function runindicator_Callback(hObject, eventdata, handles)
% hObject    handle to runindicator (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of runindicator as text
%        str2double(get(hObject,'String')) returns contents of runindicator as a double


% --- Executes during object creation, after setting all properties.
function runindicator_CreateFcn(hObject, eventdata, handles)
% hObject    handle to runindicator (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





function optpower_Callback(hObject, eventdata, handles)
% hObject    handle to optpower (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of optpower as text
%        str2double(get(hObject,'String')) returns contents of optpower as a double


% --- Executes during object creation, after setting all properties.
function optpower_CreateFcn(hObject, eventdata, handles)
% hObject    handle to optpower (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in togglebutton2.
function togglebutton2_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton2


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1


% --- Executes on button press in noelectrfilter.
function noelectrfilter_Callback(hObject, eventdata, handles)
% hObject    handle to noelectrfilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of noelectrfilter



function [pulse_train,t_vec]=Transmitter(modformat,Ppeak,rex,bitrate,no_bits,chirp,Tfwhm)


sample_bit=32;  %Number of samples per bit
Modformat=upper(modformat); %Capitalize the string 'modformat' 
t_bitslot=1/bitrate;    %The length of one bit slot

bit_sequence=[round(rand(1,no_bits-1)) 0]; %Generates a binary vector to transmit
%bit_sequence([2:2:no_bits])=ones(1,no_bits/2);

tvec=linspace(0,no_bits*t_bitslot,no_bits*sample_bit+1);  %Generate the time vector to the pulse train
t_vec=tvec(1:no_bits*sample_bit);

P1=Ppeak;   %Power level for one bits
P0=rex*P1;  %Power level for zero bits
Ppp=P1-P0;  %Peak-to-peak power

%******* NRZ modulation **************
switch Modformat
case 'NRZ'
    
    %***** Create rised-cosine pulse shape *****
    tedge=pi*linspace(0,1,floor(0.7*sample_bit)); %rise and fall time
    rise_edge=0.5*(1-cos(tedge));                 %rising edge of pulse
    top=ones(1,sample_bit-(floor(0.7*sample_bit))); %top of pulse
    fall_edge=0.5*(cos(tedge)+1);                 %falling edge of pulse
    
    pulseNRZ=[rise_edge top fall_edge];     %Single NRZ pulse (chirp free)
    
    %***** Create pulse train **************
    Pulse_train=zeros(1,sample_bit*(no_bits+1)+length(tedge)); %Vector with pulse train length containing zeros
    for k=1:no_bits
        %Add pulses
        Pulse_train(sample_bit*(k-1)+1:sample_bit*k+length(tedge))=Pulse_train(sample_bit*(k-1)+1:sample_bit*k+length(tedge))+pulseNRZ*bit_sequence(k);
    end
    pulse_train=sqrt(Ppp*Pulse_train(1:sample_bit*(no_bits))+P0);   %Output optical field 
    
    
%****** RZ ***********
case 'RZ'
    
    % *** Create Gaussian pulse shape **********************
    T0=Tfwhm/(2*sqrt(log(2)));  %Sets pulse width
    pulseRZ=(sqrt(P1)-sqrt(P0))*exp(-(1+1i*chirp)/2*((t_vec(1:sample_bit)-t_vec(sample_bit/2))/(T0)).^2);

    %***** Create pulse train **************
    pulse_train=zeros(1,sample_bit*no_bits); %Vector with pulse train length containing zeros
    for k=1:no_bits
        %Add pulses
        pulse_train(sample_bit*(k-1)+1:sample_bit*k)=pulse_train(sample_bit*(k-1)+1:sample_bit*k)+pulseRZ*bit_sequence(k);
    end
    pulse_train=pulse_train+sqrt(P0);   %Output optical field 
    
otherwise
    'Unknown modulation format';
    
end


function [pulse_train_out,Ssp_out]=FiberPropagation2(pulse_train_in,Ssp_in,alpha,D,S,L,time_vec,lambda)
% [pulse_train_out,Ssp_out]=FiberPropagation(pulse_train_in,Ssp_in,alpha,D,S,L,time_vec,lambda)
%

c=3e8;      %Speed of light in vacuum
N=length(time_vec); %Number of samples
df=1/(time_vec(end)-time_vec(1));  %Frequency resolution
omega=2*pi*(-(N/2)*df:df:(N/2-1)*df);   %Frequency vector related to the pulse train time vector 
disp=D+S*(1e9*2*pi*c./(-omega+2*pi*c/lambda)-1550); %Calculated the wavelength dependent dispersion
beta2=-disp*1e-6*lambda^2/(2*pi*c);  %Calculate beta2 from the dispersion parameter D

Fpulse_train=fftshift(fft(pulse_train_in)); %Fourier transform the pulse train

Fpulse_trainZ=Fpulse_train.*exp(1i*beta2*L.*omega.^2/2); %Add dispersion induced phase shift

pulse_train_out=ifft(ifftshift(Fpulse_trainZ))*sqrt(exp(-alpha/4.343e3*L)); %Transform back and add fiber loss

Ssp_out=Ssp_in*exp(-alpha/4.343e3*L);   % Fiber loss on noise



function [As_out,Ssp_out,G]=EDFA(As_in,Ssp_in,lambda)

Gstart=500;    %Initially gain used when finding the gain numerically  

hv=6.63e-34*3e8/lambda; %Photon energy
nsp=2;       %population-inversion factor

Pin=mean(abs(As_in).^2)+Ssp_in*30*125e9; %The total optical power into the EDFA 
Gmax=1000;     %Unsaturated gain
Psat=10e-3;    %Saturation power

err=@(G)abs(1+Psat/Pin*log(Gmax/G)-G);  %'err' should equal zero for correct gain
G=fminsearch(err,Gstart); %Numerically obtains the EDFA gain

%G=fminsearch('EDFAGain',Gstart,[],Pin); %Numerically obtains the EDFA gain

Ssp_out=2*(G-1)*nsp*hv+G*Ssp_in;    %The noise spectral density out from the EDFA 

As_out=sqrt(G)*As_in;       %The amplified signal out from the EDFA 


function [I_out,Is]=Receiver(data_in,Resp,time_vec,dv_opt,R,Ssp,forder,filterbw)


% N=length(time_vec); %Number of samples
df=1/(time_vec(2)-time_vec(1));  %Bandwidth of receiver
fbw=filterbw/(df/2);  %Electrical filter bandwidth (normalized to the half sampling rate)
q=1.602e-19;    %Electron charge
Id=1e-10;       %Receiver dark current
Fn=4;           %Receiver noise figure
kBT=1.38e-23*300;   %Boltzmann constant multiplied by room temperature

Ps_in=abs(data_in).^2;    %Received signal power

s2_t=(4*kBT/R)*Fn*df;   %signal variance due to thermal noise
s2_s=2*q*(Resp*(Ps_in+Ssp*dv_opt)+Id)*df;   %signal variance due to shot noise
s2_sig_sp=4*Resp^2*Ps_in*Ssp/2*df;          %signal variance due to signal-spontaneous beat noise
s2_sp_sp=4*Resp^2*(Ssp/2)^2*dv_opt*df;      %signal variance due to spontaneous-spontaneous beat noise

s2_tot=s2_t+s2_s+s2_sig_sp+s2_sp_sp;   %total signal variance

Is=Ps_in*Resp+sqrt(s2_tot).*randn(1,length(Ps_in));     %Detected signal with noise

[b,a]=butter(forder,fbw);   %Initialize Butterworth filter
I_out=filter(b,a,Is);       %Filters the detected signal with the Butterworth filter


function [Q,Qdata,SNR]=Qvalue(Is,no_bits,t_d,I_d)


sample_bit=32;         %Number of samples per bit

qdata=reshape(Is,sample_bit,no_bits);     %Creates a matrix (32 x number of bits) with a bit in each row
Qdata=qdata(t_d,:);             %Extracts the received signal for each bit at the decision time t_d
ONE=Qdata(Qdata>I_d);     %Finds the received 'ones' (signal > decision level)
ZERO=Qdata(Qdata<I_d);   %Finds the received 'zeros' (signal < decision level)
Q=(mean(ONE)-mean(ZERO))/(std(ONE)+std(ZERO)); %Calculates the Q-value according to p. 164 in Agrawal
SNR=20*log10((mean(ONE)-mean(ZERO))/std(ONE));
