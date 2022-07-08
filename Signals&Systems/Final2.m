close all
omega=-3:0.001:3;
s=1j*omega;

H1=(s.^2+1)./(s+1);
H2=1./(s+1);
H3=s.^2;
H4=(8*s.^2+1)./(8*s.^2-1);

plot(omega,abs(H1),'LineWidth',2)
hold on
plot([0,0],[-0.2,3],'k','LineWidth',2)
hold on
plot([-3,3],[0,0],'k','LineWidth',2)
xticks([])
yticks([])
figure

plot(omega,abs(H2),'LineWidth',2)
hold on
plot([0,0],[-0.2,1.1],'k','LineWidth',2)
hold on
plot([-3,3],[0,0],'k','LineWidth',2)
xticks([])
yticks([])
figure

plot(omega,abs(H3),'LineWidth',2)
hold on
plot([0,0],[-0.2,9],'k','LineWidth',2)
hold on
plot([-3,3],[0,0],'k','LineWidth',2)
xticks([])
yticks([])
figure

plot(omega,abs(H4),'LineWidth',2)
hold on
plot([0,0],[-0.2,1],'k','LineWidth',2)
hold on
plot([-3,3],[0,0],'k','LineWidth',2)
xticks([])
yticks([])
% figure