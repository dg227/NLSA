function lim = getUpperBatchLimit( obj, varargin );
% GETUPPERBATCHLIMIT  Get upper batch limits of nlsaPartition object 
%
% Modified 2019/08/27

lim = getBatchLimit( obj, varargin{ : } );
lim = lim( :, 2 ); 
