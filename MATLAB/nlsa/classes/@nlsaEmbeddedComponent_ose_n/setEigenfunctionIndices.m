function obj = getEigenfunctionIndices( obj, idxPhi )
% SETEIGENFUNCTIONINDICES Set the eigenfunction indices of an 
% nlsaEmbeddedComponent_ose_n object
%
% Modified 2014/06/20

if ~obj.isValidIdx( idxPhi )
    error( 'Invalid eigenfunction index specification' )
end

obj.idxPhi = idxPhi;
