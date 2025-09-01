function SetYTickLabel (ax, dxp, labels)
% h = SetYTickLabel(ax, [dxp], labels)
% Set Y-tick labels.
% This function sets Y-tick labels using text strings. The
% labels can include LaTeX-type strings. Each label is
% centered beside the corresponding Y-tick value.
% ax     - axis handle (defaults to current axis)
% dxp    - distance from the side of the graph to the edge
%          of the labels (in points), default 0.5 times the
%          font size.
% labels - string, array of strings or cell array containing 
%          the labels
%
% Example: SetYTickLabel (gca, '0|\pi|2\pi');
%          SetYTickLabel (['0   '; '\pi '; '2\pi']);
%          SetYTickLabel ({'0'; '\pi'; '2\pi'});

% $Id: SetYTickLabel.m,v 1.16 2008/09/16 03:20:20 pkabal Exp $

if (nargin == 1)
  labels = ax;
  ax = gca;
  FontSizePt = Get_FontSize_Pt(ax);
  dxp = 0.5 * FontSizePt;
elseif (nargin == 2)
  labels = dxp;
  FontSizePt = Get_FontSize_Pt(ax);
  dxp = 0.5 * FontSizePt;
end

% Note:
%   Invoking this routine a second time on a given axis does NOT
%   erase the first set of tick labels.

%===========
% Create a cell array of strings for the labels
if (iscell(labels))
  lab = labels;        % Cell array of strings

elseif (size(labels, 1) > 1)
  lab = cellstr(labels); % String array; convert to cell array

else
  rem = labels;        % Single string; look for separators
  n = 0;
  lab = {};
  while (length(rem) > 0)
    n = n+1;
    [lab{n},rem] = strtok(rem, '|'); % Pick off pieces
    if (length(rem) > 1)
      rem(1) = [];     % Remove the '|' character
    end
  end

end

%==========
% Find the conversion from points to data units
SF = SFdataXpt(ax);

YTick = get(ax, 'YTick');	% Tick values
XLim = get(ax, 'XLim');
YLim = get(ax, 'YLim');
NL = length(lab);

% Y axis labels on the right-hand side
if (strcmp(get(ax, 'YAxisLocation'), 'right'))
  
% Put text strings to the right of the tick marks at (xt, yt)
  if (strcmp(get (ax, 'XScale'), 'log'))
    xt = XLim(2) * 10^(dxp * SF(1));
  else
    xt = XLim(2) + dxp * SF(1);
  end

  W = 0;
  for i = (1:length(YTick))
    yt = YTick(i);
    if (yt >= YLim(1) && yt <= YLim(2))
      n = mod(i-1, NL) + 1;
      h = text(xt, yt, lab(n), ...
	           'VerticalAlignment', 'middle', ...
	           'HorizontalAlignment', 'left');
      W = max(W, Get_StrLen_Pt (h));       % Find the maximum string length
    end
  end

else
  
% Put text strings to the left of the tick marks at (xt, yt)
  if (strcmp(get(ax, 'XScale'), 'log'))
    xt = XLim(1) * 10^(-dxp * SF(1));
  else
    xt = XLim(1) - dxp * SF(1);
  end

  W = 0;
  for i = (1:length(YTick))
    yt = YTick(i);
    if (yt >= YLim(1) && yt <= YLim(2))
      n = mod(i-1, NL) + 1;
      h = text(xt, yt, lab(n), ...
	           'VerticalAlignment', 'middle', ...
	           'HorizontalAlignment', 'right');
      W = max(W, Get_StrLen_Pt (h));
    end
  end

  % Make a dummy YTickLabel filled with blanks
  % The goal is to fool ylabel into positioning itself to the
  % left of the Y-axis labels.
  % In 10pt Times, each blank is 3.15 pt wide.
  WBlank = 0.315;       % Normalized width of a blank
  NBlank = ceil(W / (WBlank * Get_FontSize_Pt (ax)));
  set(ax, 'YTickLabel', repmat(' ', 1, NBlank));

  % There is a problem that shows up as a difference in spacing
  % between the screen display and PS print of the plot.
  % - The spacing generated above is appropriate for the screen display,
  %   but is too crowded for the PS version.
  % - The problem shows up with a label of the form '-5\pi/2'.
  % - The spacing can be set manually with
  %     set (gca, 'YTickLabel', '      ');  % Use blanks as needed
end

return

% -----
function [WPt, HPt] = Get_StrLen_Pt (h)
% Returns the width and height of the text object in points

Units = get(h, 'Units');

set(h, 'Units', 'points');
Extent = get(h, 'Extent');

set(h, 'Units', Units);

WPt = Extent(3);
HPt = Extent(4);

return

% -----
function FontSizePt = Get_FontSize_Pt (h)
% Returns the font size in points

FUnits = get(h, 'FontUnits');

set(h, 'FontUnits', 'points');
FontSizePt = get(h, 'FontSize');

set(h, 'FontUnits', FUnits);

return
