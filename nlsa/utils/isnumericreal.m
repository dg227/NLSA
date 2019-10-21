function t = isnumericreal( x )

% isnumericreal returns logical 1 if x is numeric and real (i.e. return
% false for strings, where isreal would have returned true ).
%
% Last modified 11/26/2006
%
% t = isnumericreal( x )

t = isnumeric( x ) & isreal( x );