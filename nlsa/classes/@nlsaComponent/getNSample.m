function nS = getNSample( obj, varargin )
% GETNSAMPLE  Get number of samples in an nlsaComponent object
%
% Modified  2017/07/21

nS = getNSample( getPartition( obj ), varargin{ : } );

