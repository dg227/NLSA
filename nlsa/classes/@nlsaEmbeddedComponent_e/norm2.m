function x2 = norm2( obj, x )
% NORM2 Squared norm of embedded data in explicit storage format
% 
% Modified 2014/04/05

if size( x, 1 ) ~= getEmbeddingSpaceDimension( obj )
    error( 'Incompatible data dimension' )
end
x2 = sum( x .^ 2, 1 );
