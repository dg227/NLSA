function [ pVal, pInd ] = getOperator( obj, iB, iR, varargin )
% GETOPERATOR Read operator data of an nlsaDiffusionOperator_batch 
% object
%
% Modified 2014/12/15

Opt.precV = 'double';
Opt.precI = 'double';
Opt = parseargs( Opt, varargin{ : } );

if ~isnumeric( zeros( Opt.precV ) ) || ~isnumeric( zeros( Opt.precI ) )
    error( 'Invalid precision specification' )
end

partition = getPartition( obj );
nBTot     = getNTotalBatch( partition );

if nargin < 2 || isempty( iB )
    iB = 1 : nBTot;
end

if nargin < 3 || isempty( iR )
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

varNames = { 'pVal' 'pInd' };
if isscalar( iB )
    file = fullfile( getOperatorPath( obj ), ...
                     getOperatorFile( obj, iB, iR ) );
    load( file, varNames{ 1 : nargout } )
    eval( [ 'pVal = ' Opt.precV '( pVal );' ] ); 
    if nargout > 1
        eval( [ 'pInd = ' Opt.precI '( pInd );' ] );
    end
else
    partitionG = mergePartitions( partition );
    nS = sum( getBatchSize( partitionG, iBG ) );
    pVal   = zeros( nS, getNNeighbors( obj ), Opt.precV );
    if nargout > 1
        pInd = zeros( nS, getNNeighbors( obj ), Opt.precI );
    end
    iS1 = 1;
    for i = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( i ) ), iB( i ) ) - 1;
        file = fullfile( getOperatorPath( obj ), ...
                         getOperatorFile( obj, iB( i ), iR( i ) ) );
        B = load( file, varNames{ 1 : nargout } );
        pVal( iS1 : iS2, : ) = B.pVal;
        if nargout > 1
            pInd( iS1 : iS2, : ) = B.pInd;
        end
        iS1 = iS2 + 1;
    end
end
