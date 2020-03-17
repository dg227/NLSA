function x = getData_std( obj, iB, iR, iC, iA )
% GETDATA_STD  Read data from an array of nlsaEmbeddedComponent_o objects 
% using the standard calling syntax
%
% Modified 2020/02/17

% Validate input arguments, assign default values
siz = size( obj );
if ~isCompatible( obj ) || numel( siz ) > 3 
    error( 'First argument must be an array of compatible nlsaEmbeddedComponent objects of at most rank 3' )
end


partition  = getPartition( obj );
nBTot      = getNTotalBatch( partition( 1, : ) );

if nargin < 2 || isempty( iB )
    iB = 1 : nBTot;
end

if nargin < 3 || isempty( iR )
    iBG = iB;
    [ iB, iR ] = gl2loc( partition, iBG );
elseif nargin >= 3
    iBG = loc2gl( partition, iB, iR );
end

if nargin < 4 || isempty( iC )
    iC = 1 : size( obj, 1 );
end

if nargin < 5 || isempty( iA )
    iA = 1;
end

if numel( siz ) < 3
    siz = [ siz 1 ];
end

if ~ispsi( iA ) || iA > siz( 3 )
    error( 'Page index argument iA must be a positive scalar integer less than or equal to the page dimension of the first input argument.' )
end

if ~ispi( iR ) || any( iR > siz( 2 ) )
    error( 'Realization index argument iR must be a vector of positive integerss less than or equal to the column dimension of the first input argument.' )
end

if ~ispi( iC ) || any( iC > siz( 1 ) )
    error( 'Component index argument iC must be a vector of positive integers less than or equal to the column dimensino of the first input argument.' )
end

if ~isscalar( iR ) && ~isequal( size( iB ), size( iR ) )
    error( 'Batch index argument iB must have the same size as the realization argument iR whenever iR is non-scalar.' )
end

if ~ispi( iB ) || any( iB > getNBatch( partition( iR ) ) )
    error( 'Batch index argument iB must be a vector of positive integers less than or equal to the batch number for the corresponding realization index argument iR.' )
end

if isscalar( iB ) && isscalar( iC )
    idxE = getEmbeddingIndices( obj( iC  ) );
    file = fullfile( getDataPath( obj( iC, iR ) ), ...
                     getDataFile( obj( iC, iR ), iB ) ) ;
    load( file, 'x' )
    x = lembed( x, [ idxE( end ) size( x, 2 ) ], idxE );
else
    partitionG = mergePartitions( partition );
    nS = sum( getBatchSize( partitionG, iBG ) );
    nD = getDataSpaceDimension( obj( iC, 1 ) );
    nDTot = sum( nD );
    x = zeros( nDTot, nS );
    iS1 = 1;
    for j = 1 : numel( iB )
        iS2 = iS1 + getBatchSize( partition( iR( j ) ), iB( j ) ) - 1;
        iD1 = 1;
        for i = 1 : numel( iC )
            iD2 = iD1 + nD( iC ) - 1;
            idxE = getEmbeddingIndices( obj( iC( i ) ) );
            file = fullfile( getDataPath( obj( iC( i ), iR( j ) ) ), ...
                             getDataFile( obj( iC( i ), iR( j ) ), iB( j ) ) );
            B = load( file, 'x' );
            B.x = lembed( B.x, [ idxE( end ) size( B.x, 2 ) ], idxE );
            x( iD1 : iD2, iS1 : iS2 ) = B.x;
            iD1 = iD2 + 1;
        end
        iS1 = iS2 + 1;
    end
end

