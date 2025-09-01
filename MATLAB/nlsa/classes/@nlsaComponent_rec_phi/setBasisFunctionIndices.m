function obj = setBasisFunctionIndices( obj, idxPhi )
% SETEIGENFUNCTIONINDICES Set the eigenfunction indices of an 
% nlsaComponent_rec_phi object
%
% Modified 2015/08/28

if ~obj.isValidIdx( idxPhi )
    error( 'Invalid basis function index specification' )
end

obj.idxPhi = idxPhi;
