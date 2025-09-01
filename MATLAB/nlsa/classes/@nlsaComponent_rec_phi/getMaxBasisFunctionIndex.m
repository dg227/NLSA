function idxMax = getMaxBasisFunctionIndex( obj )
% GETMAXBASISFUNCTIONINDEX Returns the maximum basis function indices of an 
% array of nlsaComponent_rec_phi objects
%
% Modified 2015/08/28

idxMax = zeros( size( obj ) );

for iObj = 1 : numel( obj )
    idxMax( iObj ) = max( getBasisFunctionIndices( obj( iObj ) ) );
end
