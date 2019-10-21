
clear all;
close all;

%l-18 periodic
%l-22 chaos
%l-30

N = 64;  L =22;  h = 0.25;  nstp = 1000;
a0 = zeros(N-2,1);  a0(1:4) = 0.6; % just some initial condition
[tt, at] = ksfmstp(a0, L, h, nstp, 1);
fig1 = figure('pos',[5 550 600 400],'color','w'); plot(tt,at,'.-');
title('Solution of Kuramoto-Sivashinsky equation with L = 22: Fourier modes');
xlabel('Time'); ylabel('Fourier modes')

[x, ut] = ksfm2real(at, L);
fig2 = figure('pos',[5 270 600 200],'color','w');
pcolor(tt,x,ut); shading interp; caxis([-3 3]);
title('Solution u(x,t) of Kuramoto-Sivashinsky equation, system size L = 22:');
xlabel('Time'); ylabel('x','rotat',0);