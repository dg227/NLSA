function obj2 = repvecCol( obj1, src, varargin )
%% REPVEC_COL Duplicate nlsaComponent objets by repetition of column vectors
%
% Modified 2019/11/15

% Check input sizes
if ~iscolumn( obj1 ) 
    error( 'First argument must be a column vector' )
end

nC = numel( obj1 );
if ~( ismatrix( src ) && size( src, 1 ) == nC )
    error( 'Second argument must be a matrix with equal number of rows to the number of elements of the first argument' )
end

if ~iscompatible( src )
    error( 'Incompatible source components' )
end

nR = size( src, 2 );
obj2 = repmat( obj1, [ 1 nR ] );
obj2 = duplicate( obj2, src, varargin{ : } ); 


