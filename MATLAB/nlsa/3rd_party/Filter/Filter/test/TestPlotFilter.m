function TestPlotFilter

addpath('..');

% FIR filter
N = 120;
h = hamming(N);
TP(h, 1);

% IIR filter
[b, a] = ellip(5, 1, 50, 0.5);
TP(b, a);

return

%------------
% Test Plots
function TP (b, a)

figure;
PlotFilter(b, a, 'linear');
title('Linear');

figure
PlotFilter(b, a, 'log');
title('Log');

figure
PlotFilter(b, a, 'real');
title('Real');

figure
PlotFilter(b, a, 'phase-radian');
title('Phase');

figure
PlotFilter(b, a, 'delay');
title('Delay');

return
