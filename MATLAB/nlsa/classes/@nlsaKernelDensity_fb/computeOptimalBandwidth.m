function [ epsilonOpt, Info ] = computeOptimalBandwidth( obj )
% COMPUTEOPTIMALBANDWIDTH Compute optimal kernel bandwidth 
% 
% Modified 2015/09/10

nS = getNTotalSample( obj );
logE = getBandwidths( obj );
logT  = log( getDoubleSum( obj ) / nS^2 );
if isscalar( logE )
    epsilonOpt = logE;
    logE = log( logE ); 
    iOpt = [];
    dEst = [];
    dLogT = [];
    return
end

logE = log( logE  );
dLogT =  ( logT( 3 : end ) - logT( 1 : end - 2 ) ) ...
      ./ ( logE( 3 : end ) - logE( 1 : end - 2 ) );
[ dEst, iOpt ] = max( dLogT );
epsilonOpt = exp( logE( iOpt + 1 ) );

Info = struct( 'iOpt',  iOpt, ...
               'dEst',  dEst, ...
               'logE',  logE, ...
               'logT',  logT, ...
               'dLogT', dLogT );
