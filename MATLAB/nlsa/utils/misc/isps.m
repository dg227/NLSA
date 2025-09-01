function i = isps( x )
% ISPS Returns true if x is a positive scalar and false otherwise
%
% Modified 2014/03/29

i =  isscalar( x ) ...
  && x > 0 ;
