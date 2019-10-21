function nax = BlankAxes ()
% Create a new set of blank axes at the same position as the
% existing ones. The X and Y limits are set to the default values
% for a new set of axes.
% 
% $Id: BlankAxes.m,v 1.4 2008/05/01 21:07:26 pkabal Exp $

nax = copyobj(gca, gcf);
delete(get(nax, 'Children'));

% No titles, labels or grid for the new axes
axes(nax);	% New axes on top
xlabel('');
ylabel('');
zlabel('');
title('');

% Set the Position box
SetPlotPosition(nax);

% The new axes are on top, so we make the plot area transparent
% so the original plot shows through
% The new axes handle visibility is set to off so that gca finds
% the old axes
% N.B., The new axes must remain on top; issuing axes(gca)
%       brings the old axes to the top.
set(nax, 'Color', 'none');
set(nax, 'HandleVisibility', 'off');
set(nax, 'Visible', 'on');

set(nax, 'XTick', [], 'XTickLabel', []);
set(nax, 'YTick', [], 'YTickLabel', []);
set(nax, 'ZTick', [], 'ZTickLabel', []);

return

% ----- -----
function SetPlotPosition (h)
% Generate an absolute Position box compatible with the PlotBoxAspectRatio
% setting. This routine is for 2-D plots only.
%
% If PlotBoxAspectRatioMode is 'manual', the axes of the plot are placed
% inside the Position box, with the ratio of the axis lengths being
% determined by PlotBoxAspectRatio. The Position box does not reflect
% the actual size of the axes.
%
% This routine modifies the Position box to reflect the true size of the
% axes and resets PlotBoxAspectRatioMode to 'auto'.
%  (a) Work with absolute sizes (change Units temporarily)
%  (b) Scale the Position box lengths to fit within the original Position
%      box
%  (c) Center the plot within the original Position box. One axis will
%      be full size; the other will be scaled to fit and centered.
%  (d) Set PlotBoxAspectRatioMode to 'auto'
%  (f) Set PlotBoxAspectRatio to [1 1 1] - the default value. These
%      values are ignored since PlotBoxAspectRatioMode is 'auto'.

if (nargin < 1)
  h = gca;
end

PBARMode = get(h, 'PlotBoxAspectRatioMode');
if (strcmp(PBARMode, 'manual') == 1)
  PBAR = get(h, 'PlotBoxAspectRatio');
  rx = PBAR(1);
  ry = PBAR(2);
  Units = get(h, 'Units');
  set(h, 'Units', 'points');
  PosPt = get(gca, 'Position');
  x0 = PosPt(1);
  y0 = PosPt(2);
  Lx = PosPt(3);
  Ly = PosPt(4);

  % Choose a where a is the largest number such than
  %   rx * a <= Lx and ry * a <= Ly
  % a <= min(Lx/rx, Ly/ry)
  % Center the plot within the original position rectangle
  % Notes
  %  - For Matlab 6.5 this gives an axis exactly on top of the previous
  %    axes
  %  - For Matlab 7.6, this axis appears on screen to be off by about 1/2
  %    LineWidth from the previous axes. The print version of the plot
  %    is fine.
  if (Lx*ry < Ly*rx)
    Lxnew = Lx;
    Lynew = Lx * ry / rx;
    x0new = x0;
    y0new = y0 + 0.5 * (Ly - Lynew);
  else
    Lxnew = Ly * rx / ry;
    Lynew = Ly;
    x0new = x0 + 0.5 * (Lx - Lxnew);
    y0new = y0;
  end

  set(h, 'PlotBoxAspectRatio', [1 1 1]);
  set(h, 'PlotBoxAspectRatioMode', 'auto');
  set(h, 'Position', [x0new, y0new, Lxnew, Lynew]);
  set(h, 'Units', Units);

end

return
