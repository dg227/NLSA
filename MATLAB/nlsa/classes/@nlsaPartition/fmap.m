function b = fmap( obj, f, a )
% FMAP  Apply function f to elementwise batches of an array a based on an   
% nlsaPartition object obj. 
%
% Modified 2021/07/09

% Base case: a is a column vector
if iscolumn( a )
    idx = getBatchLimit( obj );
    nB  = size( idx, 1 );

    b = zeros( nB, 1 );
    for iB = 1 : nB
        b( iB ) = f( a( idx( iB, : ) ) ); 
    end
    return
end

% Recursive call if a is an array with > 2 dimensions
if ~iscolumn( a )
    siz = size( a );
    m = siz( 1 );
    n = prod( siz( 2 : end ) ); 
    a = reshape( a, [ m n ] ); 
    nB = getNBatch( obj );

    b = zeros( nB, n );
    for j = 1 : n 
        b( :, j ) = fmap( f, a( :, j ) );
    end
    b = reshape( b, [ nB siz( 2 : end ) ] );
end
