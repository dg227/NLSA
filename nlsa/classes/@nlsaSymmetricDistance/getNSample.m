function nS = getNSample( obj )
% GETNSAMPLE  Get the number of samples in each realization for an 
% nlsaSymmetricDistance object
%
% Modified 2014/04/03

nS = getNSample( getPartition( obj ) );
