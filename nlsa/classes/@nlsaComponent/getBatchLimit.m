function lim = getBatchLimit( obj, varargin );
% GETBATCHLIMIT  Get batch limits of nlsaComponent object 
%
% Modified 2017/07/20

partition = getPartition( obj );
lim = getBatchLimit( partition, varargin{ : } );
