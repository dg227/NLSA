function p = getExponent(obj, iC)
% GETEXPONENT  Get exponent of an nlsaLocalScaling_pwr object
%
% Modified 2022/11/06

p  = obj.p;

if nargin == 2 && ~isscalar(p)
    p = p(iC);
end
