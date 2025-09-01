function nS = getNSample( obj )
% GETNSAMPLE  Get the number of samples in each realization for an 
% nlsaKernelOperator object
%
% Modified 2014/07/16

nS = getNSample( getPartition( obj ) );
