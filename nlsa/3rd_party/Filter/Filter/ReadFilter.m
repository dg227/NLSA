function [b, a] = ReadFilter (FileName)
% ReadFilter: Read a TSP filter file
%   [b, a] = ReadFilter (FileName)
%   b, a: Numerator, denominator polynomials

% $Id: ReadFilter.m,v 1.3 2009/06/02 15:48:07 pkabal Exp $
fid = fopen (FileName);            % Open the file
if (fid == -1)
   error ('File not found or access denied');
end

nl = 0;
Type = [];
Cof = [];

while (1)
  line = fgetl (fid);
  if (~ischar (line))
    break;
  end
  nl = nl + 1;

  if (length(line) > 0 && line(1) == '!')
    if (nl == 1)
      Type = line;
      if (length (Type) > 4)
        Type = Type(1:4);
      end
    end
  else
  line = strrep (line, ',', ' ');  % Change commas to spaces
  Cof = [Cof str2num(line)];       % Convert to numeric form
  end

end
fclose (fid);

%==========
% Convert to direct form filter
if (strcmp (Type, '!IIR'))
  ncof = length (Cof);
  if (rem (ncof,5) ~= 0)
    error ('IIR filter: No. coefficients must be a multiple of 5');
  end
  nsect = ncof/5;
  sos5 = reshape (Cof, 5, nsect)';
  sos6 = [sos5(:,1:3), ones(nsect,1), sos5(:,4:5)];
  [b, a] = sos2tf (sos6);
  disp (['IIR filter file: ', FileName]);
  disp (['  Number of sections: ', num2str(nsect)]);

elseif (strcmp (Type, '!WIN'))
  b = Cof;
  a = 1;
  disp (['Window file: ', FileName]);
  disp (['  Number of coefficients: ', num2str(length(Cof))]);

elseif (strcmp (Type, '!FIR'))
  b = Cof;
  a = 1;
  disp (['FIR filter file: ', FileName]);
  disp (['  Number of coefficients: ', num2str(length(Cof))]);

elseif (strcmp (Type, '!ALL'))
  b = 1;
  a = Cof;
  disp (['All-pole filter file: ', FileName]);
  disp (['  Number of coefficients: ', num2str(length(Cof))]);

else
  error ('ReadFilter: Unrecognized filter type');

end
