function i = isrowstr( x )
% ISPSI Returns true if x is a row vector of characters
%
% Modified 2014/04/06

i =  ( isrow( x ) || isempty( x ) ) ...
  && ischar( x );

