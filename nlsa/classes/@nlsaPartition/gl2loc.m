function [ idxB, idxR ] = gl2loc( partition, idxG )
% GL2LOC Global to local batch index conversion 
%
% Modified 2014/06/13


if isscalar( partition )
    if ~ispi( idxG ) || any( idxG ) > getNBatch( partition )
        error( 'Invalid batch specification' )
    end
    idxB = idxG;
    idxR = ones( size( idxB ) );

elseif isvector( partition )
    idxB = zeros( size( idxG ) );
    idxR = zeros( size( idxG ) );
    nGB1 = 0; 
    for iR = 1 : numel( partition )
        nGB2 = nGB1 + getNBatch( partition( iR ) );
        ifR = idxG >= nGB1 + 1 & idxG <= nGB2;
        idxR( ifR ) = iR;
        idxB( ifR ) = idxG( ifR ) - nGB1;
        nGB1 = nGB2;
    end
end

