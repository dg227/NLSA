function idxMax = getMaxBasisFunctionIndex( obj )
% GETMAXEIGENFUNCTIONINDICES Returns the maximum basis function indices of an 
% array of nlsaEmbeddedComponent_rec objects
%
% Modified 2014/07/07

idxMax = zeros( size( obj ) );

for iObj = 1 : numel( obj )
    idxMax( iObj ) = max( getBasisFunctionIndices( obj( iObj ) ) );
end
