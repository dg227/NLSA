function nS = getNSample( obj, varargin )
% GETNSAMPLE  Get the number of samples in an nlsaKernelDensity object
%
% Modified 2017/07/20

nS = getNSample( getPartition( obj ), varargin{ : } );

