function [ obj idxG ] = mergePartitions( src );
% MERGEPARTITIONS Merge an array of nlsaPartition objects into a single nlsaPartition 
%
% Modified 2012/12/14

nSrc = numel( src );     % number of source objects
nB   = getNBatch( src ); % number of matches in the source partitions 
nBM  = sum( nB );        % number of batches in the merged partition

idx  = zeros( 1, nBM );
idxG = zeros( 2, nBM ); 

iShift = 0;
iStart = 1;
for iSrc = 1 : nSrc
    iEnd = iStart + nB( iSrc ) - 1;
    idx( iStart : iEnd )     = getIdx( src( iSrc ) ) + iShift;
    idxG( 1, iStart : iEnd ) = iSrc;
    idxG( 2, iStart : iEnd ) = 1 : nB( iSrc ); 
    iShift = iShift + getNSample( src( iSrc ) );
    iStart = iEnd + 1;
end

obj = nlsaPartition( 'idx', idx );
