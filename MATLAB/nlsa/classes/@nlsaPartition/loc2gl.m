function idxG = loc2gl( partition, idxB, idxR );
% LOC2GL Global to local batch index conversion 
%
% Modified 2017/12/23

if numel( size( idxB ) ) ~= numel( size( idxR ) ) ...
  || any( size( idxB ) ~= size( idxR ) )
    error( 'Incompatible batch and realization specification' )
end

idxG = zeros( size( idxB ) );

if isscalar( partition )
    if any( idxR ~= 1 )
        error( 'Invalid realization specification' )
    end
    if ~ispi( idxB ) || any( idxB ) > getNBatch( partition )
        error( 'Invalid batch specification' )
    end
    idxG = idxB;
elseif isvector( partition )
    nB = getNBatch( partition );
    nBAdd = cumsum( nB( 1 : end - 1 ) ); 
    ifAdd = idxR > 1;
    idxG = idxB;
    idxG( ifAdd ) = idxG( ifAdd ) + nBAdd( idxR( ifAdd ) - 1 );
end

