function [ pVal, pInd ] = computeOperator( obj, distO, diffOp, varargin )
% COMPUTEOPERATOR Compute out-of-sample extension (OSE) operator from
% query (out-of-sample) distance values
% 
% Modified 2019/07/13

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
if ~isa( distO, 'nlsaPairwiseDistance' )
    error( 'Query (out-of-sample) distance data must be specified as an nlsaPairwiseDistance object' )
end
if ~isa( diffOp, 'nlsaDiffusionOperator' )
    error( 'Third input argument must be an nlsaDiffusionOperator object' )
end
partitionO = getPartition( obj ); % partition for out-of-sample data
partition  = getPartitionTest( obj ); % partition for in-sample data
if any( ~isequal( partitionO, getPartition( distO ) ) )
    error( 'Incompatible partition of query samples' )
end
if any( ~isequal( partition, getPartition( diffOp ) ) )
    error( 'Incompatible partition of test samples' )
end
nN  = getNNeighbors( obj );
if nN > getNNeighbors( distO )
    error( 'Number of nearest neighbors in the diffusion operator cannot exceed the number of nearest neighbors in the pairwise distances' )
end
ifPrune = ~( nN == getNNeighbors( distO ) );
[ partitionG, idxG ] = mergePartitions( partitionO ); % global partition
nR       = numel( partitionO );
epsilon  = getBandwidth( obj ) * getBandwidth( diffOp );
alpha    = getAlpha( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.batch              = 1 : getNBatch( partitionG );
Opt.logFile            = '';
Opt.logPath            = obj.path;
Opt.logFilePermissions = 'w';
Opt.ifWriteOperator    = true;
Opt.ifFixZero          = true;
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
fprintf( logId, 'computeOperator starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Number of query (out-) samples = %i, \n', getNTotalSample( partitionO ) );
fprintf( logId, 'Number of test (in-) samples   = %i, \n', getNTotalSample( partition ) );
fprintf( logId, 'OSE bandwidth                  = %2.4f, \n', epsilon );
fprintf( logId, 'OSE DM normalization (alpha)   = %2.4f, \n', alpha );
fprintf( logId, 'OSE nearest neighbors          = %i, \n', nN );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the normalization and degree vectors
if alpha ~= 0
    tic
    q = getNormalization( diffOp )' .^ alpha;
    tWall = toc;
    fprintf( logId, 'READQ %2.4f \n', tWall );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the global batches
for iBG = Opt.batch

    iR = idxG( 1, iBG );
    iB = idxG( 2, iBG );
    nBR = getNBatch( partitionO( iR ) );

    tic
    [ pVal, pInd ] = getDistances( distO, iB, iR );
    if ifPrune
        pVal = pVal( :, 1 : nN );
        pInd = pInd( :, 1 : nN );
    end
    tWall = toc;
    fprintf( logId, 'READK %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 

    tic
    pVal = exp( - pVal / epsilon ^ 2 );
    if Opt.ifFixZero
        % Distances are stored in ascending order so if yVal( i, 1 ) == 0, 
        % then yVal( i, j ) == 0 for all j
        iFix = find( pVal( :, 1 ) == 0 ); 
        Opt.ifFixZero = any( iFix );
    end
    tWall = toc;
    fprintf( logId, 'EXPK %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 

    % Perform normalization by normalization vector
    if alpha ~= 0
        tic
        % Read out-of-sample normalization vector
        qO = getNormalization( obj, iB, iR ) .^ alpha;
        % q has already been raised to the power alpha
        pVal = pVal ./ q( pInd ); 
        pVal = bsxfun( @ldivide, qO, pVal ); 
        tWall = toc;
        fprintf( logId, 'NORMALIZEQ %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 
    end
  
    % Perform Markov normalization by out-of-sample degree vector
    tic
    dO = getDegree( obj, iB, iR );
    pVal = bsxfun( @ldivide, dO, pVal );
    tWall = toc;
    fprintf( logId, 'NORMALIZED %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 

    if Opt.ifFixZero
        tic
        % Set to Dirac distribution centered at the nearest neighbor
        pVal( iFix, 1 ) = 1;
        pVal( iFix, 2 : end ) = 0;
        tWall = toc;
        fprintf( logId, 'FIXZERO %i/%i %i/%i %i samples %2.4f \n', iR, nR, iB, nBR, numel( iFix ), tWall ); 
    end

    if Opt.ifWriteOperator
        tic
        setOperator( obj, pVal, pInd, iB, iR, '-v7.3' )
        tWall = toc;
        fprintf( logId, 'WRITEP %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 
    end
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeOperator finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
