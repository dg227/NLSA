function idxMax = getMaxEigenfunctionIndex( obj )
% GETMAXEIGENFUNCTIONINDICES Returns the maximum eigenfunction indices of an 
% array of nlsaEmbeddedComponent_ose_n objects
%
% Modified 2014/06/19

idxMax = zeros( size( obj ) );

for iObj = 1 : numel( obj )
    idxMax( iObj ) = max( getEigenfunctionIndices( obj( iObj ) ) );
end
