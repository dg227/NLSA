function i = isps( x )
% ISPS Returns true if x is a real scalar and false otherwise
%
% Modified 2014/04/03

i =  isscalar( x ) ...
  && isreal( x );
