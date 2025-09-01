function obj = setNNeighbors( obj, nN )
% SETNNEIGHBORS  Set number of nearest neighrors in nlsaDiffusionOperator_batch
% object
%
% Modified 2016/01/25

if ~ispsi( nN )
    error( 'The number of nearest neighbors must be a positive scalar integer' )
end
obj.nN = nN;
