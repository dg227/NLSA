function SetXTickLabel (ax, dyp, labels)
% h = SetXTickLabel(ax, [dyp], labels)
% Set X-tick labels.
% This function sets X-tick labels using text strings. The
% labels can include LaTeX-type strings. Each label is
% centered below the corresponding X-tick value.
% ax     - axis handle (defaults to current axis)
% dyp    - distance from the bottom of the graph to the top
%          of the labels (in points), default 0.5 times the
%          font size.
% labels - string, array of strings or cell array containing 
%          the labels
%
% Example: SetXTickLabel (gca, '0|\pi|2\pi');
%          SetXTickLabel (['0   '; '\pi '; '2\pi']);
%          SetXTickLabel ({'0'; '\pi'; '2\pi'});

% $Id: SetXTickLabel.m,v 1.15 2008/06/02 12:45:00 pkabal Exp $

if (nargin == 1)
  labels = ax;
  ax = gca;
  FontSizePt = Get_FontSize_Pt(ax);
  dyp = 0.5 * FontSizePt;
elseif (nargin == 2)
  labels = dyp;
  FontSizePt = Get_FontSize_Pt(ax);
  dyp = 0.5 * FontSizePt;
end

% Note:
%   Invoking this routine a second time on a given axis does NOT
%   erase the first set of tick labels.

%===========
% Create a cell array of strings for the labels
if (iscell(labels))
  lab = labels;           % Cell array of strings

elseif (size(labels, 1) > 1)
  lab = cellstr(labels); % String array; convert to cell array

else
  rem = labels;           % Single string; look for separators
  n = 0;
  lab = {};
  while(length(rem) > 0)
    n = n+1;
    [lab{n},rem] = strtok(rem, '|'); % Pick off pieces
    if (length(rem) > 1)
      rem(1) = [];        % Remove the '|' character
    end
  end

end

%==========
set(ax, 'XTickLabel', ' '); % Remove normal tick labels
                             % Leave a blank, so xlabel
                             % is positioned properly

% Find the conversion from points to data units
SF = SFdataXpt(ax);

% Put text strings below the tick marks at (x, y)
XTick = get(ax, 'XTick');	% Tick values
XLim = get(ax, 'XLim');
YLim = get(ax, 'YLim');

if (strcmp(get(ax, 'YScale'), 'log'))
  yt = YLim(1) * 10^(-dyp * SF(2));
else
  yt = YLim(1) - dyp * SF(2);
end

NL = length(lab);

for i = (1:length(XTick))
  xt = XTick(i);
  if (xt >= XLim(1) && xt <= XLim(2))
    n = mod(i-1, NL) + 1;
    text(xt, yt, lab{n}, ...
      	 'VerticalAlignment', 'cap', ...
	     'HorizontalAlignment', 'center');
  end
end

return

% -----
function FontSizePt = Get_FontSize_Pt (h)
% Returns the font size in points

FUnits = get(h, 'FontUnits');

set(h, 'FontUnits', 'points');
FontSizePt = get(h, 'FontSize');

set(h, 'FontUnits', FUnits);

return
