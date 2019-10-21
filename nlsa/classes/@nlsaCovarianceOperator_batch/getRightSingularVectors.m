function [ v, s ] = getRightSingularVectors( obj, iB, iR )
% GETRIGHTSINGULARVECTORS  Read right singular vectors of an 
% nlsaCovarianceOperator object
%
% Modified 2014/07/16

partition  = getPartition( obj );
nBTot      = getNTotalBatch( partition );

if nargin < 2
    iB = 1 : nBTot;
end

if nargin < 3
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
end

varNames = { 'v' };
if isscalar( iB )
    file = fullfile( getRightSingularVectorPath( obj ), ...
                     getRightSingularVectorFile( obj, iB, iR ) );
    load( file, varNames{ : } )
else
    partitionG = mergePartitions( partition );
    nS = sum( getBatchSize( partitionG, iBG ) );
    v = zeros( nS, getNEigenfunction( obj ) );
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getRightSingularVectorPath( obj ), ...
                         getRightSingularVectorFile( obj, iB( i ), iR( i ) ) );
        B = load( file, varNames{ 1 : min( nargout, 2 ) } );
        v( iS1 : iS2, : ) = B.v;
        iS1 = iS2 + 1;
    end
end

if nargout == 2
    s = getSingularValues( obj );
end
