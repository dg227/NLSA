function iB = findBatch( obj, iS, varargin )
% FINDBATCH Find batch indices of samples in an nlsaKernelDensity object
%
% Modified 2017/07/20

partition = getPartition( obj );
iB = findBatch( partition, iS, varargin{ : } );
