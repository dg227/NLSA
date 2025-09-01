function Tick = GenTicks (v, NTickR)
% Generate a set of ticks for use in a linear scale plot.
% v - Data vector used to determine the data range.
% NTickR - Number of ticks desired. This value is a suggestion. The actual
%     number of ticks will be chosen by this routine.
% Tick - Tick values
%
% The the first tick is placed at or below Vmin and the last tick is placed
% at or above Vmax. The tick spacing is a number of the form 1, 2 or 5
% times a power of ten.

% The goal is to produce a set of ticks that agrees with what the Matlab
% plot function would choose. The agreement is good except in some cases
% which plot seems to choose unreasonable and/or inconsistent values.
%
% The number of ticks depends on the size of the plot. The following
% results are for the default plot size.
% (1) data [0 1.1]
%     - plot gives YTick = 0:0.2:1.4
% (2) data [ 0 11]
%     - plot gives YTick = 0:2:12
% (3) data [1.0001 1.0002]
%     - plot in Matlab 6.5 gives 11 ticks
%     - plot in Matlab 7.1/7.2 gives 3 ticks
% (4) data [-1.0002 -1.0001]
%     - 11 ticks for plot
% Notes:
% - Case (1) generates an extra tick compared with case (2).
% - One would expect case (3) and case (4) to have the same number of
%   ticks.
% - Note that generating 11 ticks for case (3) or (4) will result in
%   repeated tick labels since the interval between ticks is too small to
%   be distinguishable with the precision alloted to the labels.
% - This routine reduces the number of ticks for cases such as (3) or (4)
%   so that the tick labels differ.

% $Id: GenTicks.m,v 1.5 2009/06/02 16:25:19 pkabal Exp $

% Example:
% - Data range vMax-Vmin is 10, then the power of 10 scaling is 1 (for
%   NTickR = 10) and the scaled step sizes are [0.1 0.2 0.5 1 2 5 10 20
%   50].
% - Assuming the end points are multiples of the step size, the numbers of
%   intervals are [100 50 20 10 5 2 1 1 1].
% - If the end points are just a bit outside of multiples of the step size,
%   the numbers of intervals increase to [102 52 22 12 7 4 3 2 2].

% Normalized tick spacing
DeltaN = [ 0.1 0.2 0.5 1 2 5 10 20 50 ];

% Desired number of ticks and penalty for deviation
% Some clues as to how plot chooses its axes can be found in
% the routine plotyy. There NickOpt is 6 and Penalty is 0.02.
if (nargin < 2)
  NTickR = 10;
end
Penalty = 0.05;
eDelN = 1e-4;

% Find the min and max values
% If the data is constant, choose a scale
% If there are no finite values, the returned tick vector will also be
% empty.
Vmin = min (v(isfinite(v)));
Vmax = max (v(isfinite(v)));
if (Vmin == Vmax)
  Vmin = Vmin - 1;
  Vmax = Vmax + 1;
end

% Return for empty limits
if (isempty (Vmin))
  Tick = [];
  return;
end

% Data span
Vdel = Vmax - Vmin;

% Adjust the number of ticks if the spacing between ticks is small.
% - Ticks are labelled (by plot) with a maximum of 5 digits,
%   e.g. 1.0001 or -1.0001. With this resolution, only step
%   sizes larger than about 0.0001 will result in different
%   tick labels.
% The number of steps is chosen as follows,
%   (N-1) Del >= Vmax - Vmin
%           N >= (Vmax - Vmin) / Del + 1
% We want to establish a lower limit on Del
%   Del >= eDel
% Let
%   Va = max (|Vmax|, |Vmin|).
% Express Va as
%   Va = VaN P
% where P = 10^m, and m = floor(log10(Va)). Then VaN is a normalized
% value, 1 <= VaN < 10. It is assumed that VaN will be a label and the
% power of ten will be a separate annotation on the axis. Then we can
% denormalize eDelN,
%   eDel = eDelN P.
% Finally,
%   N >= (Vmax - Vmin) / (P * eDelN) + 1.
Va = max (abs(Vmax), abs(Vmin));
P = 10^floor(log10(Va));
NTickmax = Vdel / (P * eDelN) + 1;
NTickR = min (NTickmax, NTickR);
  
% Scale the normalized spacing values
% Find a power of 10 to scale the data
P10 = 10^round (log10(Vdel/NTickR));
DeltaS =  P10 * DeltaN;

% Find a set of upper and lower bounds which are a multiple of the spacings
VU = DeltaS .* ceil (Vmax ./ DeltaS);
VL = DeltaS .* floor (Vmin ./ DeltaS);

% Number of ticks for each possible spacing
% VU - VL is a multiple of DeltaS; the round is just to protect against
% computational errors
NTick = round ((VU - VL) ./ DeltaS) + 1;

% Goodness of fit
% - First term measures the graph occupied by the data
% - Second term assigns a penalty for deviating from NTickR tick values
%   (no additional penalty for +/1 one deviation)
vfit = ((VU - VL) - Vdel) ./ (VU - VL);
fit = vfit + Penalty * max (NTick - NTickR, 1).^2;

% Find the combination with the best fit
% There are many ways to calculate Tick from VL, VU and DeltaS.
% The method here gives exactly the same values as used by plot
[fitopt, i] = min (fit);
Tick = VL(i) + (0:NTick(i)-1) * DeltaS(i);

return
