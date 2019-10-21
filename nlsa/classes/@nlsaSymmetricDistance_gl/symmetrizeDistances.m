function symmetrizeDistances( obj, dist, varargin )
% SYMMETRIZEDISTANCES Compute symmetric distance matrix from pairwise distances
% 
% Modified 2014/01/15

if ~isa( dist, 'nlsaPairwiseDistance' )
    error( 'Distance data must be specified as an nlsaPairwiseDistance object' )
end

nR   = getNRealization( dist );
nS   = sum( getNSample( dist ) );
nB   = getNBatch( dist );
nNIn = getNNeighbors( dist );
nN   = getNNeighbors( obj );

if nN > nNIn
    error( 'Number of nearest neighbors in symmetrized distance cannot exceed the number of source nearest neighbors' )
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
Opt.logFile = '';
Opt.logPath = obj.path;
Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'w' );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'symmetrizeDistances starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of realizations             = %i \n', nR );
fprintf( logId, 'Total number of samples            = %i \n', nS );
fprintf( logId, 'Total number of batches            = %i \n', sum( nB ) );
fprintf( logId, 'Number of input nearest neighbors  = %i \n', nNIn );
fprintf( logId, 'Number of output nearest neighbors = %i \n', nN );

% Allocate distance arrays
tic
yVal = zeros( nS * nN, 1 );
yCol = zeros( nS * nN, 1 );
yRow = zeros( nS * nN, 1 );
tWall = toc;
fprintf( logId, 'ALLOCATE %i nonzero elements \n', nS * nN, tWall );

% Merge realization partitions to form a global partition
partition = mergePartitions( getPartition( dist ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import the pairwise distance batches
iBG = 1; % Global batch index
for iR = 1 : nR
    for iB = 1 : nB( iR )
    
        % j are column indices
        % indStart, indEnd are linear indices
        jLim    = getBatchLimit( partition, iBG );   
        indLim( 2 ) = jLim( 2 ) * nN;
        indLim( 1 ) = ( jLim( 1 ) - 1 ) * nN + 1;      
        nSB         = getBatchSize( dist, iB, iR );
 
        [ yValSrc, yIndSrc ] = getDistances( dist, iB, iR );
        tWall = toc;
        fprintf( logId, 'READ realization %i/%i, batch %i/%i, samples %i-%i (%i samples) %2.4f \n', ...
                 iR, nR, iB, nB, jLim( 1 ), jLim( 2 ), nSB, tWall );

        tic  
        yValSrc         = yValSrc( :, 1 : nN )'; % trim nearest neighbors    
        yIndSrc         = yIndSrc( :, 1 : nN )';
        yVal( indLim( 1 ) : indLim( 2 ) ) = reshape( yValSrc, [ nN * nSB, 1 ] );
        yCol( indLim( 1 ) : indLim( 2 ) ) = double( reshape( yIndSrc, [ nN * nSB, 1 ] ) );
        tWall = toc;
        fprintf( logId, 'TRIM realization %i/%i, batch %i/%i, %i -> %i nearest neighbors %2.4f \n', iR, nR, iB, nB, nNIn, nN, tWall );
        iBG = iBG + 1;
    end
end
clear yValSrc yIndSrc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the nonzero entries in the non-symmetric distances 

tic
yRow   = ones( nN, 1 ) * ( 1 : nS );
yRow   = reshape( yRow, nS * nN, 1 );
ifZero = yVal < 1E-6;
yRowNZ = yRow( ~ifZero );
yColNZ = yCol( ~ifZero );
yValNZ = sqrt( yVal( ~ifZero ) );
nNZ    = numel( yRowNZ );
clear yVal
yRow = yRow( ifZero );
yCol = yCol( ifZero );
nZ   = numel( yRow );
clear ifZero
tWall = toc;

fprintf( logId, '----------------------------------------------------------------, \n' );
fprintf( logId, 'ZEROSCAN (non-symmetric distances) %2.4f \n', tWall );
fprintf( logId, 'Total number of entries    = %i \n', nS * nN );
fprintf( logId, 'Number of nonzero elements = %i \n', nNZ );
fprintf( logId, 'Number of zero elements    = %i (%i samples ) \n', nZ, nS ); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Symmetrize distances using an "OR" operation

% Nonzero elements
tic
y = sparse( yRowNZ, yColNZ, yValNZ, nS, nS, nNZ );
clear yRowNZ yColNZ yValNZ
y2 = y .* y.'; % y2 contains the squares of the distances
y  = y .^ 2;
y  = y + y' - y2;
clear y2 % preserve memory
t = toc;
[ yRowNZ yColNZ yValNZ ] = find( y );
nNZ = nnz( y );  
tWall = toc;

fprintf( logId, 'SYMMETRIZATION (nonzero entries) %2.4f \n', tWall );
fprintf( logId, 'Number of nonzero elements = %i \n', nNZ );
                                      
% Zero elements
y    = sparse( yRow, yCol, ones( nZ, 1 ), nS, nS, nZ );
y2   = y .* y.';
y    = y + y' - y2;
clear y2 % preserve memory
[ yRow yCol yVal ] = find( y );
yVal( 1 : end ) = 0;
nZ = nnz( y );
tWall = toc;

fprintf( logId, 'SYMMETRIZATION (zero entries) %2.4f \n', tWall );
fprintf( logId, 'Number of zero elements = %i \n', nZ );

tic
yRow = int32( [ yRow; yRowNZ ] ); % preserve memory and disk space
clear yRowNZ
yCol = int32( [ yCol; yColNZ ] );
clear yColNZ
yVal = [ yVal; yValNZ ];
clear yValNZ
tWall = toc;
fprintf( logId, 'MERGE %i nonzero entries, %i zero entries %2.4f \n', nNZ, nZ, tWall );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save calculation results
tic                                      
setDistances( obj, yRow, yCol, yVal, '-v7.3' ) % support large files
tWall = toc;
fprintf( logId, 'WRITE %i samples, %i nearest neighbors, %2.4f average nearest neighbors per sample %2.4f \n', nS, nN, numel( yVal ) / nN, tWall );

clk = clock; % Exit gracefully
fprintf( logId, 'symmetrizeDistances finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
                                    
                                   
