function h = MidTick (varargin)
% h = MidTick ([[ax,] Axes])
% Generate intermediate ticks on a plot axis.
% This function generates intermediate ticks between the existing X, Y, or
% Z axis ticks in a plot. For a linear axis, the intermediate ticks (0.75
% of the normal length) lie midway between the existing ticks. For a log
% axis, the ticks (0.5 of the normal length) lie at 1, 2, ..., 9 times the
% decade points. The intermediate ticks are unlabelled.
%
% The arguments specify the plot axes which get the mid-ticks. The axes
% arguments are given in the order X, Y, and Z. If an axis argument is
% empty, no mid-ticks are placed on the axis. If the axis argument is the
% string 'X', 'Y' or 'Z' (as appropriate), mid-ticks are placed at the
% default locations for that axis. If the axis argument is a vector, the
% intermediate ticks are placed at those locations.
% (1) Specify the axes to have default X and Y mid-ticks.
%     h = MidTick('X', 'Y');  or  h = MidTick('XY');
% (2) Specify the axes to have mid-ticks only for the Y axis.
%     h = MidTick('Y');
% (3) Specify the intermediate tick locations for the Y axis
%     h = MidTick([], [0.5 1.5]);
% (4) Default tick locations for the X axis and specify the intermediate
%     tick locations for the Y axis.
%     h = MidTick('X', [0.5 1.5]);
% (5) Override the intermediate ticks. The routine generates the default
%     mid-ticks, the locations of which are then overridden with a plot
%     option.
%     h = MidTick('Y', 'YTick', [0.5, 1.5]);
%     
% For all cases, the arguments can be preceded by an axis handle (gca by
% default). Also, additional trailing arguments can be used to specify
% other plot options, such as 'TickLength'.
%
% Note that Matlab generates minor ticks for log axes. The only reason to
% have this routine do so for log axes is for those cases where some of the
% minor ticks are missing due to a bug in Matlab (for versions up to 7.2 at
% least).
%
%  A new transparent axis is generated on top of the existing axis. Note
%  that this new axis must remain on top for the newly generated ticks to
%  be visible. The handle for this new axis is returned.
%  *** Invoke this routine after all plotting and labelling is done ***
%  *** It is suggested that the graph limits be set (using a call to
%      'axis', for instance) before invoking this routine ***

% $Id: MidTick.m,v 1.27 2008/09/17 11:49:08 pkabal Exp $

[ax, XTmSpec, YTmSpec, ZTmSpec, varg] = ProcArgs (varargin);
axes(ax);       % Make ax the current axis

% Generate the intermediate tick values
[XTickm FreezeX] = GenTickm(XTmSpec, 'X');
[YTickm FreezeY] = GenTickm(YTmSpec, 'Y');
[ZTickm FreezeZ] = GenTickm(ZTmSpec, 'Z');

% New axes - TickLength defined
h = SetTickmLength();

% Set tick locations
SetTickm(h, XTickm, YTickm, ZTickm);

% Apply the rest of the arguments
if (~isempty(varg))
  set(h, varg{:});
end

Freeze = [FreezeX FreezeY FreezeZ];
if (~ isempty(Freeze))
  disp(['>>> MidTick - Plot limits (', Freeze, ') frozen']);
end

return

%=========
function [Tickm Freeze] = GenTickm (TmSpec, Axis)

Tickm = [];
Freeze = [];

if (~isempty(TmSpec))
  if (isnan(TmSpec))
    Lim = get(gca, [Axis 'Lim']);    % XLim or YLim or ZLim
    Tick = get(gca, [Axis 'Tick']);
    Scale = get(gca, [Axis 'Scale']);
    Tickm = Gen_Tick(Tick, Lim, Scale);
  else
    Tickm = TmSpec;
  end
  if (~isempty(Tickm) && strcmpi(get(gca, [Axis 'LimMode']), 'auto'))
    Lim = get(gca, [Axis 'Lim']);
    set(gca, [Axis 'Lim'], Lim);
    Freeze = Axis;
  end
end

return

%==========
% Generate ticks midway between existing ticks
function htick = Gen_Tick (Tick, Lim, Scale)

NTick = length(Tick);
htick = [];

if (NTick > 1)
  if (strcmpi(Scale, 'linear'))
    htick = 0.5 * (Tick(1:NTick-1) + Tick(2:NTick));

    % See if we need to put some "mid-ticks" outside of the
    % Tick values
    dTick = diff(Tick);
    dTickMax = max(dTick);
    dTickMin = min(dTick);
    
    % Check for near-constant interval lengths
    if (dTickMax - dTickMin < 2 * eps * dTickMax)
      htickL = Tick(1) - 0.5 * dTick(1);
      if (htickL >= Lim(1))
        htick = [htickL, htick];
      elseif (htickL > Lim(1) - 2 * eps * dTickMax)
        htick = [Lim(1), htick];
      end
      htickU = Tick(end) + 0.5 * dTick(end);
      if (htickU <= Lim(2))
        htick(end+1) = htickU;
      elseif (htickU < Lim(2) + 2 * eps * dTickMax)
        htick(end+1) = Lim(2);
      end
    end
  
  else
    LTick = floor(log10(Tick(1))):floor(log10(Tick(NTick)));
    htick = [1 2 3 4 5 6 7 8 9]' * 10.^LTick;
    htick = (htick(:))';
    htick = htick (htick >= Lim(1) & htick <= Lim(2));
  end

end

return

% ==========
function h = SetTickmLength ()

% There is only one TickLength parameter for the whole plot
if (strcmpi(get(gca, 'XScale'), 'log') || ...
    strcmpi(get(gca, 'YScale'), 'log') || ...
    strcmpi(get(gca, 'ZScale'), 'log'))
  TickLengthm = 0.5 * get(gca, 'TickLength');
else
  TickLengthm = 0.75 * get(gca, 'TickLength');
end

% Put the mid-ticks into a new axis
h = BlankAxes;
set(h, 'TickLength', TickLengthm);

return

% ==========
function SetTickm (h, XTickm, YTickm, ZTickm)

set(h, 'XTick', XTickm);
if (length(XTickm) > 1 )
  set(h, 'XMinorTick', 'off');
end
set(h, 'YTick', YTickm);
if (length(YTickm) > 1)
  set(h, 'YMinorTick', 'off');
end
set(h, 'ZTick', ZTickm);
if (length(ZTickm) > 1)
  set(h, 'ZMinorTick', 'off');
end

return

%==========
% Process arguments, filling in defaults
function [ax, XTmSpec, YTmSpec, ZTmSpec, options] = ProcArgs (varg)

% Axis, default to gca
ax = gca;
if (~isempty(varg) && ishandle(varg(1)))
  ax = varg{1};
  varg(1) = [];
end

% Expand concatenated axes specs and decode specs
varg = ExpXYZ(varg, 'XY');
varg = ExpXYZ(varg, 'XYZ');
varg = ExpXYZ(varg, 'YZ');
[XTmSpec, varg] = DecXYZ('X', varg);
varg = ExpXYZ(varg, 'YZ');
[YTmSpec, varg] = DecXYZ('Y', varg);
[ZTmSpec, varg] = DecXYZ('Z', varg);

% Rest of the arguments
options = varg;

return

% =========
function varg = ExpXYZ (varg, Axes)
% Expand 'XYZ' to {'X' 'Y' 'Z'} in the argument list

if (~isempty(varg) && strcmp(varg(1), Axes))
  N = length(Axes);
  for (i = 1:N)
    C{i} = Axes(i);
  end
  varg(1) = [];
  varg = {C{:} varg{:}};
end

return

% =========
function [TmSpec, varg] = DecXYZ (Axis, varg)
% Process an axis argument, Axis is 'X', 'Y', or 'Z'

% TmSpec on output is [], NaN, or a vector (no ticks, default ticks,
% specified ticks)
% varg on output has had the processed argument removed.

TmSpec = [];
if (~isempty(varg))
  if (isempty(varg{1}))
    % No mid-ticks
    varg(1) = [];     % Remove the argument
  elseif (strcmp(varg{1}, Axis))
    % Default mid-ticks
    TmSpec = NaN;
    varg(1) = [];
  elseif (isnumeric(varg{1}))
    % Specified intermediate ticks
    TmSpec = varg{1};
    varg(1) = [];
  end
end

return
