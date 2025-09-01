function SymbolLabels (Axis, Sym, SF, Mode)
% Add fractional symbol labels for the plot axes
%  SymbolLabels (Axis, Sym, SF, Mode)
%  Axis - 'X' or 'Y' to choose the plot axes to be labelled
%  Sym - Numerator symbol, for instance '\pi' or '1/T'.
%  SF - Scale factor for the axis. The axis tick values will be scaled by
%       this value. The default value is 1.
%  Mode - Mode, either 'Fraction' (default) or 'Decimal'
%
% Fraction Mode:
% The input string Sym will be decomposed into NSym/Dsym at the "/".
% After scaling by SF, the axis tick value "1" will get labelled as
% NSym/DSym. If the text interpreter is "tex" (the default for Matlab),
% the Sym string can contain symbols such as '\pi' or '{\itN}'. LaTeX
% syntax can be used for the Sym value if the text interpreter is set to
% "latex".
%
% Examples:
% 1) Axis ticks: [0, 0.25, 0.5, 0.75, 1] with Sym = 'P/Q'.
%    Tick labels: '0', 'P/(4Q)', 'P/(2Q)', '3P/(4Q)'. 'P/Q'.
% 2) Axis ticks: [0, 1, 2, 3, 4], with Sym = '1/Q' and SF = 0.5.
%    Tick labels: '0', '1/(2Q)', '1/Q', '3/(2Q)', '2/Q'. 
% 3) Axis ticks" [-0.5, 0, 0.5, 1, 1.5] with Sym = ''.
%    Tick labels will be '-1/2', '0', '1/2', '1', '3/2', '2'.
%
% Decimal Mode:
% The input string symbol will be appended to the existing tick label.
%
% Examples
% 1) Axis ticks: [0, 0.25, 0.5, 0.75, 1] with Sym = 'Q'.
%    Tick labels: '0', '0.25Q', '0.5Q', '0.75Q'. 'Q'.
% 2) Axis ticks: [0, 1, 2, 3, 4], with Sym = '1/Q' and SF = 0.5.
%    Tick labels: '0', '0.5/Q', '1/Q', '1.5/Q', '2/Q'.

% $Id: SymbolLabels.m,v 1.7 2008/09/25 15:18:19 pkabal Exp $

Fraction = 1;
Decimal = 2;

if (nargin == 1)
  Sym = '';
  SF = 1;
  Mode = 'Fraction';
elseif (nargin == 2)
  if (ischar(Sym))
    SF = 1;
  else
    SF = Sym;
    Sym = '';
  end
  Mode = 'Fraction';
elseif (nargin == 3)
  if (ischar(SF))
    Mode = SF;
    SF = 1;
  else
    Mode = 'Fraction';
  end
end

switch Axis
  case 'X'
    Tick = get(gca, 'XTick');
  case 'Y'
    Tick = get(gca, 'YTick');
end

IMode = strmatch(lower(Mode), {'fraction', 'decimal'});

LString = FracSym(SF*Tick, Sym, IMode);

switch Axis
  case 'X'
    SetXTickLabel(LString);
  case 'Y'
    SetYTickLabel(LString);
end

return

% ----------
function LString = FracSym (V, Sym, IMode)
% Return a string with a fraction given by Sym. 
% Examples for Sym = 'S/T', with IMode set to fraction or decimal
%    V         LString
%  -0.5    -S/(2T)  -0.5S/T
%   0      0         0
%   0.5    S/(2T)    0.5S/T
%   1      S/T       S/T
%   1.5    3S/(2T)   1.5S/T
%   2      2S/T      2S/T

Fraction = 1;
Decimal = 2;

[NSym, DSym] = ParseSym(Sym);
if (strcmp(NSym, '1'))
  NSym = '';
end

LString = [];
NV = length (V);
for (i = 1:NV)

  % Parse the rational approximation into a numerator and denominator
  if (IMode == Fraction)
    fracS = rats(V(i));
    [NumS, DenS] = ParseSym(fracS);
  else
    NumS = num2str(V(i));
    DenS = '';
  end

  % Numerator string
  if (strcmp(NumS, '0'))
    LString{i} = '0';
  else
    if (~isempty(NSym))
      if (strcmp(NumS, '1'))
        NumS = '';
      elseif (strcmp(NumS, '-1'))
        NumS = '-';
      end
    end
    NumSS = strcat(NumS, NSym);

    % Denominator string
    if (isempty(DenS))
      DenSS = DSym;
    elseif (~isempty(DSym))
      DenSS = strcat('(', DenS, DSym, ')');
    else
      DenSS = DenS;
    end
 
    if (isempty(DenSS))
      LString{i} = NumSS;
    else
      LString{i} = strcat(NumSS, '/', DenSS);
    end
  end

end

return

% ----------
function [NS, DS] = ParseSym (S)
% Parse 'abc/def' at the '/'

posD = findstr('/', S);
if (isempty(posD))
  NS = StringTrim(S);
  DS = '';
else
  NS = StringTrim(S(1:posD-1));
  DS = StringTrim(S(posD+1:end));
end
  
return

% ----------
function STrim = StringTrim (S)
% Trim leading and trailing blanks

[r, c] = find(~isspace (S));
if (isempty (c))
  STrim = S([]);
else
  STrim = S(:,min(c):max(c));
end

return
