function i = ispvi( x )
% ISPVI Returns true if x is a vector of positive integers and false otherwise
%
% Modified 2014/03/07

if ~isvector( x )
    i = false;
    return
end

if x <= 0
    i = false;
    return
end

if any( x ./ round( x ) ~= 1 )
    i = false;
    return
end

i = true;
