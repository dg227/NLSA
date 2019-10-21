function [ nBefore, nAfter ] = getNSampleFD( obj )
% GETNSAMPLEFD Returns the number of samples that must be kept before and after
% the current sample for finite differences
%
% Modified 2014/04/05

nFD     = getFDOrder( obj );
switch getFDType( obj )
    case 'backward'
        nBefore = nFD;
        nAfter  = 0;
    case 'central'
        nBefore = nFD / 2;
        nAfter  = nBefore;
    case 'forward'
        nBefore = 0;
        nAfter  = nFD;
end

