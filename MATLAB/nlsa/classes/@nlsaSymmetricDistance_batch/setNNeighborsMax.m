function obj = setNNeighborsMax( obj, nNMax )
% SETNNEIGHBORSMAX Set maximum number of nearest neighrbors of an 
% nlsaSymmetricDistance_batchobject
%
% Modified 2012/04/28

if ~ispsi( nNMax ) || nNMax < getNNeighbors( obj )
    error( 'Maximum Number of nearest neighbors must be a positive scalar integer greater or equal than the number of nearest neighbors' )
end
obj.nNMax = nNMax;
