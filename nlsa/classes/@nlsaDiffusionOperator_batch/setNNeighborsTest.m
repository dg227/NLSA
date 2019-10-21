function obj = setNNeighborsTest( obj, nN )
% SETNNEIGHBORSTEST  Set number of nearest neighrors for the test data of an
%  nlsaDiffusionOperator_batch object
%
% Modified 2016/01/25

if ~ispsi( nN )
    error( 'The number of nearest neighbors must be a positive scalar integer' )
end
obj.nNT = nN;
