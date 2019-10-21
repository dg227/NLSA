function nEig = getNEigenfunction( obj )
% GETNEIGENFUNCTION Returns the number of eigenfunctions in an array of
% nlsaKernelOperator objects
%
% Modified 2014/07/16

nEig = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nEig( iObj ) = obj( iObj ).nEig;
end
