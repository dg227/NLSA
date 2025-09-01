function [SF, XScale, YScale] = SFdataXpt (ax)
% SFDataXPt - Return the conversion factor units/pt. The
% conversion factor is determined from the figure. If the
% an axis is in log units, the corresponding conversion
% factor is log10(units)/pt.
%  SF - data units / points
%  ax - Axes handle, defaults to gca

% This routine gets its information from a figure. A typical
% preamble before calling this routine is as follows.
%     SetPlotSize([xdim, ydim], 'centimeters');
%     axis([xmin, xmax, ymin, ymax]);
%     set (gca, 'XScale', 'log');  % For X log scale
%     set (gca, 'YScale', 'log');  % For Y log scale

% $Id: SFdataXpt.m,v 1.7 2006/09/27 18:15:55 pkabal Exp $

if (nargin == 0)
  ax = gca;
end

Figs = get (0, 'Children');
if (isempty (Figs))
  fprintf ('>>> Warning, axis size and limits are undefined\n');
else
  XLimMode = get (ax, 'XLimMode');
  YLimMode = get (ax, 'YLimMode');
  if (strcmp (XLimMode, 'auto') || strcmp (YLimMode, 'auto'))
    fprintf ('>>> Warning, axis limits not explicitly set\n');
  end
end

% Get the axis size in points
Units = get (ax, 'Units');
set (ax, 'Units', 'points');
PosPt = get (ax, 'Position');
set (ax, 'Units', Units);  % Restore the axes units

XLim = get (ax, 'XLim');
YLim = get (ax, 'YLim');

XScale = get (ax, 'XScale');
if (strcmp (XScale, 'log'))
  XLim = log10 (XLim);
end
YScale = get (ax, 'YScale');
if (strcmp (YScale, 'log'))
  YLim = log10 (YLim);
end

SF = [diff(XLim) diff(YLim)] ./ PosPt(3:4);

return
