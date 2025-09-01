function lim = getLowerBatchLimit( obj, varargin );
% GETLOWERBATCHLIMIT  Get lower batch limits of nlsaPartition object 
%
% Modified 2019/08/27

lim = getBatchLimit( obj, varargin{ : } );
lim = lim( :, 1 ); 
