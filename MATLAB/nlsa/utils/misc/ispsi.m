function i = ispsi( x )
% ISPSI Returns true if x is a positive scalar integer and false otherwise
%
% Modified 2014/03/29

i =  isscalar( x ) ...
  && x > 0 ...
  && x / round( x ) == 1;
