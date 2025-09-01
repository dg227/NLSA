function obj = setBasisFunctionIndices( obj, idxPhi )
% SETEIGENFUNCTIONINDICES Set the eigenfunction indices of an 
% nlsaEmbeddedComponent_rec object
%
% Modified 2014/07/07

if ~obj.isValidIdx( idxPhi )
    error( 'Invalid basis function index specification' )
end

obj.idxPhi = idxPhi;
