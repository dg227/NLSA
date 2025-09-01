function i = isnnsi( x )
% ISNZSI Returns true if x is a scalar nonnegative integer and false otherwise
%
% Modified 2012/12/05

if numel( x ) > 1
    i = false;
    return;
end

if x < 0
    i = false;
    return;
end

if x ~= 0 && x / round( x ) ~= 1
    i = false;
    return;
end

i = true;
