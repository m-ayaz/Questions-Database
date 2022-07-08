clear
clc
%% 15dB NRZ
Q15nrz=[4.3 8.1 7.87 7.7 7.56 7.38 7.27 7.19 ...
    7.78 8.61 8.65 8.2 8.17 8.8 8.9 6];
BW15nrz=[0.4 0.7:0.05:1 0.5:0.05: 0.65 0.67 0.57 0.56 0.45];
% bar(BW15nrz,Q15nrz)
x15nrz=0.4:0.0001:1.2;
y15nrz=spline(BW15nrz,Q15nrz,x15nrz);
subplot(2,2,1)
plot(x15nrz,y15nrz,'.')
% xlim([0.45 1.2])
title('NRZ with 15dB attenuation')
xlabel('BW')
ylabel('Q')
%% 12 dB NRZ
Q15nrz=[4.63 7.1 10.96 14.01 16.5 12.9 13.12 ...
    12.6 14.8];
BW15nrz=[0.4:0.05:.75 0.57];
% bar(BW15nrz,Q15nrz)
x15nrz=0.4:0.0001:0.75;
y15nrz=spline(BW15nrz,Q15nrz,x15nrz);
subplot(2,2,2)
plot(x15nrz,y15nrz,'.')
title('NRZ with 12dB attenuation')
xlabel('BW')
ylabel('Q')
%% 15 dB RZ
Q15nrz=[4.8 4.85 5.1 4.8 4.95 5.02 5.11 5.04];
BW15nrz=[2 1.5 1 0.8 0.9 .95 1.05 1.1];
% bar(BW15nrz,Q15nrz)
x15nrz=0.8:0.0001:2;
y15nrz=spline(BW15nrz,Q15nrz,x15nrz);
subplot(2,2,3)
plot(x15nrz,y15nrz,'.')
title('RZ with 15dB attenuation')
xlabel('BW')
ylabel('Q')
%% 12 dB RZ
Q15nrz=[8.18 8.96 9.52 9.8 10.12 10.2 10.23 ...
    10.21 10.2 9.45];
BW15nrz=[0.8:0.05:1.15 1.25 2];
% bar(BW15nrz,Q15nrz)
x15nrz=0.8:0.0001:2;
y15nrz=spline(BW15nrz,Q15nrz,x15nrz);
subplot(2,2,4)
plot(x15nrz,y15nrz,'.')
title('RZ with 12dB attenuation')
xlabel('BW')
ylabel('Q')