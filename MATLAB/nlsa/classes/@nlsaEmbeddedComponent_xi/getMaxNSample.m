function maxNSE = getMaxNSample( obj, nS )
% GETMAXNSAMPLE  Get maximum number of samples after embedding in an array nlsaEmbeddedComponent_xi objects given source data of nS samples
%
% Modified 2013/12/10

maxNSE = getMaxNSample@nlsaEmbeddedComponent( obj, nS );

maxNSEFd   = 0;
fdSubtract = 0;
for iObj = 1 : numel( obj )
    switch obj( iObj ).fdType
        case 'forward'
            fdSubtract = obj( iObj ).fdOrd;
        case 'central'
            fdSubtract = obj( iObj ).fdOrd / 2;
    end
    maxNSEFd = max( maxNSEFd, fdSubtract );
end

maxNSE = maxNSE - maxNSEFd;
