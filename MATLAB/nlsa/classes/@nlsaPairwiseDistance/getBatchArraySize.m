function [ nSB, nN ] = getBatchArraySize( obj, iB, iR )
% GETBATCHARRAYSIZE  Get batch sizes of an nlsaPairwiseDistance object 
%
% Modified 2014/04/14

partition = getPartition( obj );

switch nargin
    case 1
        iB = 1 : getNTotalBatch( partition );
        [ nSB, nN ] = getBatchArraySize( obj, iB );
    case 2
        [ iB, iR ] = gl2loc( partition, iB );
        [ nSB, nN ] = getBatchArraySize( obj, iB, iR );
    case 3
        nSB = zeros( size( iB ) );
        if isscalar( iR ) && isvector( iB )
            for i = 1 : numel( iB )
                nSB( i ) = getBatchSize( partition( iR ), iB( i ) );
            end
        elseif isvector( iR ) && isvector( iB )
            if numel( iB ) ~= numel( iR )
                error( 'Incompatible batch and realization specification' )
            end
            for i = 1 : numel( iB )
                nSB( i ) = getBatchSize( partition( iR( i ) ), iB( i ) );
            end
        else
            error( 'Invalid batch and realization specification' )
        end
end

nN = getNNeighbors( obj );
        
