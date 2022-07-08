clear
clc
close all
% t=0:0.001:2*pi;
% 
% x=sin(t);
% y=cos(t);
% h=plot(x,y,':');
% h.LineWidth=2;
% axis equal
aa=15;
% [hz2, hp2, ht2] = zplane([],[0.5+0.5j;0.5-0.5j;1.5]);
% hold off;
% % set(findobj(ht1, 'Type', 'line'), 'Color', 'r');
% set(findobj(ht2, 'Type', 'line'), 'LineWidth', 2);
% set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa);
% set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa);
% % h.MarkerSize=20;
figure
[hz2, hp2, ht2] =zplane([0],[0.8]);
set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);

figure
[hz2, hp2, ht2] =zplane([],[-0.8;0.8]);
set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);

figure
[hz2, hp2, ht2] =zplane([0.8*exp(1j/1.3*pi/4);0.8*exp(-1j/1.3*pi/4)],[0.8*exp(1j*1.1*pi/4);0.8*exp(-1.1j*pi/4)]);
set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);

figure
[hz2, hp2, ht2] =zplane(0.8*exp(1j*linspace(0,7,8)*pi/4)',[]);
set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
%% %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%
figure
[hz2, hp2, ht2] =zplane([-1;0.3],[0.8;1.2]);
set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);

figure
[hz2, hp2, ht2] =zplane([0],[-1]);
set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);

figure
[hz2, hp2, ht2] =zplane([0.8;1.2],[-1;0.3]);
set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);