function dSum = computeDoubleSum( obj, dist, varargin )
% COMPUTEDOUBLESUM Compute double kernel sum from candidate bandwidths 
% 
% Modified 2015/05/07

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
if ~isa( dist, 'nlsaPairwiseDistance' )
    error( 'Distance data must be specified as an nlsaPairwiseDistance object' )
end
partition = getPartition( obj );
if any( ~isequal( partition, getPartition( dist ) ) )
    error( 'Incompatible partitions' )
end
[ partitionG, idxG ] = mergePartitions( partition ); % global partition  
nR      = numel( partition );
nS      = getNTotalSample( partition ); 
nBG     = getNBatch( partition );
epsilon = getBandwidths( obj );
nE      = numel( epsilon );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments 
Opt.batch              = 1 : getNBatch( partitionG );
Opt.logFile            = '';
Opt.logPath            = getDensityPath( obj );
Opt.logFilePermissions = 'w';
Opt = parseargs( Opt, varargin{ : } );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), ...
                   Opt.logFilePermissions );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeDoubleSum starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Number of samples              = %i, \n', nS );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the global batches
dSum = zeros( 1, nE );
for iBG = 1 : nBG

    iR  = idxG( 1, iBG );
    iB  = idxG( 2, iBG );
    nBR = getNBatch( partition( iR ) );
    iS  = getBatchLimit( partitionG, iBG );

    tic 
    y = getDistances( dist, iB, iR );
    tWall = toc;
    fprintf( logId, 'READK %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 

    tic
    for iE = 1 : nE
        dSum( iE ) = dSum( iE ) + sum( sum( exp( -y / epsilon( iE ) ^ 2 ) ) );
    end
    tWall = toc;
    fprintf( logId, 'EXP %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 
       
    tic
    setDoubleSum( obj, dSum, '-v7.3' )
    tWall = toc;
    fprintf( logId, 'WRITEDSUM %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 
end 

clk = clock; % Exit gracefully
fprintf( logId, 'computeDoubleSum finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
