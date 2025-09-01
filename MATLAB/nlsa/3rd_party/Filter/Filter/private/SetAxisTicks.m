function SetAxisTicks (axislim)
% This function is to be used after plotting to choose the axis ticks.
% The tick values differ from the default values used by plot. It is
% assumed that the plot limits have been chosen and only the tick values
% used to label the axis need be set.
% - The number of ticks is reduced (from about 10 to 6)
%
% Notes:
% - This routine does not change the axes for logarithmic axes.
% - The number of ticks depends on whether the end of the axes coincide
%   with a tick.

% $Id: SetAxisTicks.m,v 1.5 2008/05/01 18:59:49 pkabal Exp $

if (nargin < 1)
  axislim = axis;
end

XLim = axislim(1:2);
YLim = axislim(3:4);

NTickR = 6;
if (strcmp (get(gca,'XScale'), 'linear'))
  XTick = Gen_Ticks (XLim, NTickR);

% Only set the ticks if they are different from the default
  XTickP = get (gca, 'XTick');
  if (~ isempty (XTick) ...
      && (length (XTick) ~= length (XTickP) || any (XTick ~= XTickP)))
      set (gca, 'XTick', XTick);
  end
end

if (strcmp (get(gca,'YScale'), 'linear'))
  YTick = Gen_Ticks (YLim, NTickR);

% Check for a crowded Y axis
  Frac = FracLabelY (YTick, YLim);
  if (Frac > 0.8)  % Reduce the no. ticks if the Y axis is crowded
    YTick = Gen_Ticks (YLim, NTickR - 1);
    Frac = FracLabelY (YTick, YLim);
    if (Frac > 0.8)
      YTick = Gen_Ticks (YLim, NTickR - 2);
    end
  end

% Only set the ticks if they are different from the default
  YTickP = get (gca, 'YTick');
  if (~ isempty (YTick) ...
      && (length (YTick) ~= length (YTickP) || any (YTick ~= YTickP)))
    set (gca, 'YTick', YTick);
  end
end

return

% ----------
function Tick = Gen_Ticks (Lim, NTickR)

% Initial Tick values
Tick = GenTicks (Lim, NTickR);

% If both ticks occur outside of the data, increase the
% number of ticks.
%        |-------------------|     plot limits, span dV
%        X   X   X   X   X   X     labels at end points
%      X    X    X    X    X    X  labels off the end   
%
% The number of tick intervals in the plot span is
%    gA = dV / dT,
% where dV = Vu - Vl and dT = (Tu - Tl) / (N-1).
% For N ticks, this value can be bounded as gA > N-3.
% If the ticks occur at the end points of the data interval,
%    gmax = N-1.
% If the difference between gmax and gA is larger than one,
% we will try to increase the number of ticks. In fact we
% only increase the "requested number of ticks". The number of
% ticks may not actually increase.
NTick = length (Tick);
if (NTick > 1)
  gA = (NTick - 1) * (Lim(2) - Lim(1)) / (Tick(end) - Tick(1));
  if (gA < NTick - 2)
    Tick = GenTicks (Lim, NTickR + 1);
  end
end

if (Tick(1) < Lim(1))
  Tick(1) = [];
end
if (Tick(end) > Lim(2))
  Tick(end) = [];
end

return

% ----------
function Frac = FracLabelY (YTick, YLim)
% This function returns the fraction of the vertical space
% taken up by the Y tick labels. Each tick label is assumed
% to be the height of a zero character.

% The size (Extent) of a character string includes some white
% space around the character. So, when this routine reports
% Frac = 1, the white spaces just touch, i.e. Frac = 1, is
% acceptable in the sense that the characters do not actually
% overlap.

units = get (gca, 'Units');
set (gca, 'Units', 'centimeter');
pos = get (gca, 'Position');
YSize = pos(4);
set (gca, 'Units', units);

NTick = length (YTick);
gA = (NTick - 1) * (YLim(2) - YLim(1)) / (YTick(end) - YTick(1));
TickIntSize = YSize / gA;

Zpt = 0.0476;    % 10 point '0' has an Extent 0.238 by 0.476 cm
fontsize = get (gca, 'FontSize');
CharHeightSize = Zpt * fontsize;

Frac = CharHeightSize / TickIntSize;

return
