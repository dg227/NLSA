function str = idx2str( idx, prefix )

if nargin == 1
    prefix = 'idx';
end

if numel( idx ) == 1
    str = [ prefix int2str( idx ) ];
else
    idxSkip  = idx( 2 ) - idx( 1 );
    idxTrial = idx( 1 ) : idxSkip : idx( end );
        
    if numel( idx ) == numel( idxTrial ) && all( idx == idxTrial )
        str = [ prefix int2str( idx( 1 ) ) '-' ...
                '+'    int2str( idxSkip ) '-' ...
                int2str( idx( end ) ) ];
    else
        str = [ prefix sprintf( '%i-', idx ) ];
        str = str( 1 : end - 1 );
    end
end
