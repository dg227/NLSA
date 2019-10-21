function q = getNormalization( obj, iB, iR )
% GETNORMALIZATION Read normalization data of an nlsaDiffusionOperator_batch 
% object
%
% Modified 2018/06/18

partition = getPartition( obj );
nBTot     = getNTotalBatch( partition );

if nargin < 2
    iB = 1 : nBTot;
end

if nargin < 3
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
end

if isscalar( iB )
    file = fullfile( getNormalizationPath( obj ), ...
                     getNormalizationFile( obj, iB, iR ) );
    load( file, 'q' )
else
    partitionG = mergePartitions( partition );
    nS = sum( getBatchSize( partitionG, iBG ) );
    q  = zeros( nS, 1 );
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getNormalizationPath( obj ), ...
                         getNormalizationFile( obj, iB( i ), iR( i ) ) );
        B = load( file, 'q' );
        q( iS1 : iS2, : ) = B.q;
        iS1 = iS2 + 1;
    end
end
