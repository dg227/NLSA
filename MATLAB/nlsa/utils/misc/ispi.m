function i = ispi( x )
% ISPI Returns true if is an array of positive integers and false otherwise
%
% Modified 2020/04/10

if x <= 0
    i = false;
    return
end

if any( x ./ round( x ) ~= 1 )
    i = false;
    return
end

i = true;
