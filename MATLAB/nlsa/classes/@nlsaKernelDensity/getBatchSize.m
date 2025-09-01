function nSB = getBatchSize( obj, varargin )
% GETBATCHSIZE  Get batch sizes of an nlsaKernelDensity object 
%
% Modified 2017/07/20

partition = getPartition( obj );
nSB = getBatchSize( partition, varargin{ : } );
