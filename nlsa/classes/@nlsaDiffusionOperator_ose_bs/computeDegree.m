function d = computeDegree( obj, distO, diffOp, varargin )
% COMPUTEDEGREE Compute kernel degree from out-of-sample distance
% data distO. diffOp is the in-sample diffusion operator.
% 
% Modified 2018/07/08

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
if ~isa( distO, 'nlsaSymmetricDistance_batch' ) ...
  && ~isa( distO, 'nlsaPairwiseDistance' )
    error( 'Distance data must be specified as nlsaSymmetricDistance_batch or nlsaPairwiseDistance objects' )
end
partition = getPartition( obj );  % partition for query data
if any( ~isequal( partition, getPartition( distO ) ) )
    error( 'Incompatible distance partition' )
end
nN = getNNeighbors( obj );
if isa( distO, 'nlsaSymmetricDistance_batch' )
    if nN ~= getNNeighborsMax( distO )
        error( 'Incompatible number of nearest neighbors' )
    end
    ifPrune = false;
elseif isa( distO, 'nlsaPairwiseDistance' )
    if nN > getNNeighbors( distO )
        error( 'Number of nearest neighbors in the diffusion operator cannot exceed number of nearest neighbors in the pairwise distances' )
    end
    ifPrune = ~( nN == getNNeighbors( distO ) );
end
[ partitionG, idxG ] = mergePartitions( partition );
nR       = numel( partition );
epsilon  = getBandwidth( obj ) * getBandwidth( diffOp );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments 
Opt.batch              = 1 : getNBatch( partitionG );
Opt.logFile            = '';
Opt.logPath            = getOperatorPath( obj );
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
iBatch = Opt.batch;

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeDegree starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Number of samples              = %i, \n', getNTotalSample( partition ) );
fprintf( logId, 'Gaussian width                 = %2.4f, \n', epsilon );
fprintf( logId, 'Nearest neighbors              = %i, \n', nN );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the global batches
for iBG = iBatch
    iR = idxG( 1, iBG );
    iB = idxG( 2, iBG );
    nB = getNBatch( partition( iR ) );

    tic
    yVal = getDistances( distO, iB, iR );
    if ifPrune
        yVal = yVal( :, 1 : nN );
    end
    tWall = toc;
    fprintf( logId, 'READK %i/%i %i/%i %2.4f \n', iR, nR, iB, nB, tWall ); 

    tic    
    yVal = exp( - yVal / epsilon ^ 2 );
    tWall = toc;
    fprintf( logId, 'EXP %i/%i %i/%i %2.4f \n', iR, nR, iB, nB, tWall ); 

    tic
    d = sum( yVal, 2 );
    tWall = toc;
    fprintf( logId, 'SUMD %i/%i %i/%i %2.4f \n', iR, nR, iB, nB, tWall ); 
    
    tic
    setDegree( obj, d, iB, iR, '-v7.3' )
    tWall = toc;
    fprintf( logId, 'WRITED %i/%i %i/%i %2.4f \n', iR, nR, iB, nB, tWall ); 
    
end


clk = clock; % Exit gracefully
fprintf( logId, 'computeDegree finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
