function obj = setBasisFunctionIndices( obj, idxPhi )
% SETBASISFUNCTIONINDICES Set the basis function indices of an 
% nlsaKoopmanOperator object
%
% Modified 2020/04/10

if ~obj.isValidIdx( idxPhi )
    error( 'Invalid basis function index specification' )
end

obj.idxPhi = idxPhi;
