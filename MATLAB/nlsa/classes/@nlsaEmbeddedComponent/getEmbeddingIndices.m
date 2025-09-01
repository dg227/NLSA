function idxE = getEmbeddingIndices( obj )
% GETEMBEDDINGINDICES  Get embedding indices of nlsaEmbeddedComponent objects
%
% Modified 2012/04/21

if isscalar( obj )
    idxE = obj.idxE;
else
    idxE = cell( size( obj ) );
    for iObj = 1 : numel( obj )
        idxE{ iObj } = obj.idxE;
    end
end
