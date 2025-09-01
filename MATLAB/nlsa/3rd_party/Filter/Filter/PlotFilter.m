function [fp, Hp] = PlotFilter (b, a, Fs, YLim, NPlot, Option)
% Plot a filter response.
%  - 'log':    Log amplitude plot (default plot type)
%  - 'log-radian': Log plot as above, with the frequency axis labelled
%              with normalized radian frequency
%  - 'linear': Linear amplitude plot
%  - 'linear-radian': Linear amplitude plot, with the frequency axis
%              labelled with normalized radian frequency
%  - 'real':   Linear plot of the real part of the response (zero-phase
%              response). The frequency response is decomposed into the
%              product of a phase factor (the phase can be non-linear)
%              and a real factor.
%  - 'real-radian': Linear plot of the real part, with the frequency axis
%              labelled in normalized radian frequency.
%  - 'phase':  Plot of the phase factor (see above).
%  - 'phase-radian': Plot of the phase factor, with the frequency axis
%              labelled in normalized radian frequency.
%  - 'delay':  Plot of the group delay (derivative of the phase response).
%  - 'delay-radian': Plot of the group delay, with the frequency
%              axis labelled in normalized radian frequency.
%
% If the last argument is found to be a string, this is taken to be the
% option string.
%
%  b, a: Filter coefficients (numerator, denominator)
%  Fs: Sampling frequency (default 1)
%    This can also be a 3 element array, where the values are
%    interpreted to mean [Fmin, Fmax, Fs]. These values are always in
%    Hz units.
%  YLim: Y-axis limits [Ymin, Ymax]. These values are in the units of the
%    plot (dB, linear, radians, or samples as appropriate)
%  Nplot: Number of plot points (automatically determined by
%    default). The actual number of plot points may be smaller if the
%    plot is smooth.
%  Option: Plot type
%
% [fp, Hp] - output frequency values and data values
%    These values are exactly what was passed to the plot routine. These
%    values can be used to replot the data with different plot options.
%
% PlotFilter (b, a, Fs, [Option])
% PlotFilter (b, a, [Option])
% PlotFilter (b, [Option])
% Use a large number of points so that the ripples for high
% high filters are properly represented. The frequencies of the
% evaluation points are modified to better represent the envelope
% of the freqiency response. After generating the data, the
% plot vectors are merged to remove unnecessary points.

% $Id: PlotFilter.m,v 1.16 2009/06/02 15:47:47 pkabal Exp $

% Interpret the options
switch nargin
  case (6)
    [FiltCoef, Fspec, YLim, NPlot, Option, SubOption] ...
               = DecArgs(b, a, Fs, YLim, NPlot, Option);
  case (5)
    [FiltCoef, Fspec, YLim, NPlot, Option, SubOption] ...
               = DecArgs(b, a, Fs, YLim, NPlot);
  case (4)
    [FiltCoef, Fspec, YLim, NPlot, Option, SubOption] ...
               = DecArgs(b, a, Fs, YLim);
  case (3)
    [FiltCoef, Fspec, YLim, NPlot, Option, SubOption] = DecArgs(b, a, Fs);
  case (2)
    [FiltCoef, Fspec, YLim, NPlot, Option, SubOption] = DecArgs(b, a);
  case (1)
    [FiltCoef, Fspec, YLim, NPlot, Option, SubOption] = DecArgs(b);
  otherwise
    [FiltCoef, Fspec, YLim, NPlot, Option, SubOption] = DecArgs;
end

% Calculate the frequency response at adjusted frequency points
% Note that this adjustment is done on the amplitude, so the frequency
% points are adjusted to find zeros and local extrema of the response.
% This is important, not only for amplitude plots, but also for the phase
% response, since the phase can do wild excursions at the zeros.
[H, f] = AdjustFreq(FiltCoef.b, FiltCoef.a, Fspec, NPlot);
if (strcmp (Option, 'delay'))
  [H, f] = grpdelay(FiltCoef.b, FiltCoef.a, f, Fspec.Fs);
end

% Convert frequency to multiples of normalized radians
if (strcmp(SubOption, 'radian'))
  f = 2 * f / Fspec.Fs;
  Fspec.Lim = 2 * Fspec.Lim ./ Fspec.Fs;
  Fspec.Fs = 2;
end

switch Option
case {'log'}
  [fp, Hp] = dBAmplPlot(f, H, Fspec, YLim, FiltCoef);

case {'linear'}
  [fp, Hp] = LinAmplPlot(f, H, Fspec, YLim);

case {'real'}
  [fp, Hp] = RealPlot(f, H, Fspec, YLim);

case {'phase'}
  [fp, Hp] = PhasePlot(f, H, Fspec, YLim);

case ('delay')
  [fp, Hp] = DelayPlot(f, H, Fspec, YLim);

otherwise
  error('PlotFilter >> Invalid option');

end

% Add small ticks between labelled ticks
MidTick('XY');

% Add fractional pi notation for the frequency axis
if (strcmp(SubOption, 'radian'))
  SymbolLabels('X', '{\it\pi}');
end

return

% ----------
function [f, HdB] = dBAmplPlot (f, H, Fspec, YLim, FiltCoef)

HdB = 20 * log10(abs(H));

if (isempty(YLim))
  YLim = dBRange(HdB, FiltCoef);
end

% Merge points
axis([Fspec.Lim, YLim]);
[f, HdB] = XYmerge(f, HdB);

% Plot the data
plot(f, HdB);
axis([Fspec.Lim, YLim]);
SetAxisTicks([Fspec.Lim, YLim]);

return

% -----------
function [f, HA] = LinAmplPlot (f, H, Fspec, YLim)

HA = abs (H);

% Plot limits, include 0 in Y axis
if (isempty (YLim))
  YLim = GenLimits([HA(:); 0]);
end

% Merge closely spaced points
axis([Fspec.Lim, YLim]);
[f, HA] = XYmerge(f, HA);

% Plot the data
plot(f, HA);
axis([Fspec.Lim, YLim]);
SetAxisTicks([Fspec.Lim, YLim]);

return

% -----------
function [f, HR] = RealPlot (f, H, Fspec, YLim)

HR = UnwrapPi(H);

% Plot limits, include 0 in Y axis
if (isempty(YLim))
  YLim = GenLimits([HR(:); 0]);
end

% Merge closely spaced points
axis([Fspec.Lim, YLim]);
[f, HR] = XYmerge(f, HR);

% Plot the data
plot(f, HR);
axis([Fspec.Lim YLim]);
SetAxisTicks([Fspec.Lim, YLim]);

if (YLim(1) < 0 && YLim(2) > 0)
  Xaxis;
end

return

% ----------
function [f, HP] = PhasePlot (f, H, Fspec, YLim)

[HR, HP] = UnwrapPi(H);
HP = HP / pi;     % Angle in multiples of radians

% Merge closely spaced points
if (isempty(YLim))
  YLim = [floor(min([HP;0])), ceil(max([HP;0]))];
end
axis([Fspec.Lim, YLim]);
[f, HP] = XYmerge(f, HP);

plot(f, HP);
axis([Fspec.Lim, YLim]);
SetAxisTicks([Fspec.Lim, YLim]);

% Add fractional pi notation for the phase axis
SymbolLabels('Y', '{\it\pi}');

if (YLim(1) < 0 && YLim(2) > 0)
  Xaxis;
end

return

% ----------
function [f, Hd] = DelayPlot (f, Hd, Fspec, YLim)

% Excise wild excursions
I = (abs(Hd) > 1e5);
Hd(I) = NaN;

% Plot limits, include 0 in Y axis
if (isempty(YLim))
  YLim = GenLimits([Hd(:);0]);
end

% Merge closely spaced points
axis([Fspec.Lim, YLim]);
[f, Hd] = XYmerge(f, Hd);

plot(f, Hd);
axis([Fspec.Lim, YLim]);
SetAxisTicks([Fspec.Lim, YLim]);

if (YLim(1) < 0 && YLim(2) > 0)
  Xaxis;
end

return

% ----------
function YLim = dBRange (HdB, FiltCoef)

YdBrange = 60;
if (length(FiltCoef.b) > 30)
  YdBrange = 80;
elseif (length(FiltCoef.b) > 80)
  YdBrange = 100;
elseif (length(FiltCoef.a) > 100)
  YdBrange = 100;
end

% Move upper end to a multiple of 10 dB or 20 dB
% (20 dB steps result in fewer labels)
% - Choose 20 dB multiple if the space at the top
%   of the plot will still be smaller than the space
%   at the bottom of the plot
HdBmax = max(HdB);
Ymax = ceil(HdBmax / 10) * 10;
YmaxT = ceil(HdBmax / 20) * 20;
if (YmaxT ~= Ymax)
  HdBmin = min(HdB);
  if (YmaxT - HdBmax <= HdBmin - (YmaxT - YdBrange))
    Ymax = YmaxT;
  end
end

YLim = [Ymax-YdBrange, Ymax];

return

%-----------
% Decode options
function [FiltCoef, Fspec, YLim, NPlot, Option, SubOption] ...
                    = DecArgs (b, a, Fs, YLim, NPlot, Option)

switch nargin
  case (6)
  case (5)
    if (ischar(NPlot))
      Option = NPlot;
      NPlot = [];
    else
      Option = [];
    end
  case (4)
    if (ischar(YLim))
      Option = YLim;
      YLim = [];
    else
      Option = [];
    end
    NPlot = [];
  case (3)
    if (ischar(Fs))
      Option = Fs;
      Fs = [];
    else
      Option = [];
    end
    YLim = [];
    NPlot = [];
  case (2)
    if (ischar(a))
      Option = a;
      a = 1;
    else
      Option = [];
    end
    Fs = [];
    YLim = [];
    NPlot = [];
  case (1)
    a = 1;
    Fs = [];
    YLim = [];
    NPlot = [];
    Option = [];
  otherwise
    error ('PlotFilter >> Invalid number of arguments');    
end

FiltCoef.b = b;
FiltCoef.a = a;

if (isempty(Option))
  Option = 'log';
end
if (isempty(NPlot))
  NPlotMin = 501;
  NPlotMax = 4001;
  NSing = length(FiltCoef.b) - 1 + length(FiltCoef.a) - 1;
  NPlot = min(max(10 * NSing, NPlotMin), NPlotMax);
end
if (isempty(Fs))
  Fspec.Fs = 1;
  Fspec.Lim = [0, Fspec.Fs/2];
elseif (length(Fs) == 1)
  Fspec.Fs = Fs;
  Fspec.Lim = [0, Fspec.Fs/2];
elseif (length(Fs) == 2)
  Fspec.Lim = Fs;
  Fspec.Fs = 1;
elseif (length(Fs) == 3)
  Fspec.Lim= Fs(1:2);
  Fspec.Fs = Fs(3);
else
  error('PlotFilter >> Invalid frequency specification');
end
SubOption = 'Hz';
switch Option
 case {'log-radian', 'linear-radian', 'real-radian', ...
       'phase-radian', 'delay-radian'}
    Option = Option(1:end-7);
    SubOption = 'radian';
end
    
return

% -----
function VLim = GenLimits (v)
% Generate tick marks and return the end ticks

NTickR = 6;

VLim = [min(v), max(v)];

% Initial Tick values
Tick = GenTicks(VLim, NTickR);

% If the ticks occur outside of the data, increase
% the number of ticks
if (Tick(1) < VLim(1) || Tick(end) > VLim(2))
  Tick = GenTicks(VLim, NTickR + 1);
end

VLim = [Tick(1), Tick(end)];

return

% -----
function [HR, HP] = UnwrapPi (H)
% Return the zero-phase component and unwrapped phase for a densely
% sampled complex frequency response.

% The Matlab function unwrap handles jumps of 2pi. However a change
% of sign involves a jump of pi. This routine also unwraps these jumps.
 
HR = abs(H);
HP = unwrap(angle(H));
Idx = find(diff(HP) > pi/2) + 1;
for (i = Idx')
  HP(i:end) = HP(i:end) - pi;
  HR(i:end) = -HR(i:end);
end

return
