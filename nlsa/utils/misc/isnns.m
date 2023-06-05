function i = isnns(x)
% ISNNS Returns true if x is a non-negative scalar and false otherwise
%
% Modified 2023/05/08

i =  isscalar(x) ...
  && x >= 0 ;
