function obj = setBasisFunctionIndices( obj, idxPhi )
% SETBASISFUNCTIONINDICES Set the basis function indices of an nlsaLinearMap 
% object
%
% Modified 2014/07/20

if ~obj.isValidIdx( idxPhi )
    error( 'Invalid basis function index specification' )
end

obj.idxPhi = idxPhi;
