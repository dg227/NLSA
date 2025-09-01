function h = Xaxis (y, varargin)
% h = Xaxis (y, <plot options>)
%
% Generate an X axis at the given y-value
% This function generates a new (nearly) zero height axis. The new
% axis is placed on top of the existing axis.
%   - position: y-value from the input value
%               x-range from the current axis
%   - ticks: tick labels are turned off. The top and bottom
%     X-axes overlap (appearing as a single axis) with the tick
%     marks extending to either side of the axis.
%
% Additional axes property values for this axis can be specified
% in the variable length argument list.
%   - 'Box': Set it to 'off' to get ticks only up or down
%   - 'XAxisLocation: With 'Box' set to 'off', setting this to
%     'top' gives downward ticks, while 'bottom' gives upward
%     ticks

% $Id: Xaxis.m,v 1.9 2008/05/01 21:07:35 pkabal Exp $

XLim = get(gca, 'XLim');
YLim = get(gca, 'YLim');
XTick = get(gca, 'XTick');

if (nargin < 1)
  y = 0;
end

if (y < YLim(1) || y > YLim(2))
  disp('Xaxis: position off scale');
  h = [];
  return;
end

Peps = 1e-6;
yn = (y - YLim(1)) / (YLim(2) - YLim(1));
ym = yn - 0.5 * Peps;
yp = yn + 0.5 * Peps;

DL = YLim(2) - YLim(1);
YS = YLim(1);
YLim(1) = YS + ym * DL;
YLim(2) = YS + yp * DL;

% New axes on top of old ones
h = BlankAxes;

% Change the Position box
Pos  = get(h, 'Position');
Pos(2) = Pos(2) + ym * Pos(4);
Pos(4) = Peps * Pos(4);

set(h, 'Position', Pos, ...
       'Box', 'on', ...
       'XLim', XLim, ...
       'XTick', XTick, ...
       'YLim', YLim, ...
       'YTick', [], ...
       'XAxisLocation', 'bottom', ...
	   varargin{:});

return
