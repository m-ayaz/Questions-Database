close all
figure
[hz2, hp2, ht2] =zplane([-0.8],[1.2+1j;1.2-1j]);

aa=15;

set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
text(-0.8+0.08,0+0.08,'-0.8','FontSize',16)
text(1.1,1-0.13,'1.2+1j','FontSize',16)
text(1.1,-1+0.13,'1.2-1j','FontSize',16)
xticks([])
yticks([])





figure
[hz2, hp2, ht2] =zplane([],[0.5;-1.5]);
set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
text(-1.5,0+.13,'-1.5','FontSize',16)
text(.5,0.13,'0.5','FontSize',16)
% text(1.1,-1+0.1,'1.2-1j','FontSize',16)
xticks([])
yticks([])



figure
[hz2, hp2, ht2] =zplane([-1;0],[0.3;.7]);
set(findobj(ht2, 'Type', 'line'), 'LineWidth', 1,'LineStyle','-','Color','k');
set(findobj(hz2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
set(findobj(hp2, 'Type', 'line'), 'MarkerSize', aa,'LineWidth',3);
% text(-0.8+0.08,0+0.08,'-0.8','FontSize',16)
text(0.3,0.15,'0.3','FontSize',16)
text(0.7,0.15,'0.7','FontSize',16)
xticks([])
yticks([])