function [ pVal, pInd ] = computeOperator( obj, dist, varargin )
% COMPUTEOPERATOR Compute diffusion operator in batch format from distance
% data dist
% 
% Modified 2018/06/14

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
if ~isa( dist, 'nlsaSymmetricDistance_batch' )
  && ~isa( dist, 'nlsaPairwiseDistance' )
    error( 'Distance data must be specified as nlsaSymmetricDistance_batch or nlsaPairwiseDistance objects' )
end
partition = getPartition( obj );  % partition for query data
if any( ~isequal( partition, getPartition( dist ) ) )
    error( 'Incompatible distance partition' )
end
if any( ~isequal( partition, getPartition( dist ) ) )
    error( 'Incompatible partitions' )
end
[ partitionG, idxG ] = mergePartitions( partition ); % global partition  
nN = getNNeighbors( obj );
if isa( dist, 'nlsaSymmetricDistance_batch' )
    if nN ~= getNNeighborsMax( dist )
        error( 'Incompatible number of nearest neighbors' )
    end
    ifPrune = false;
elseif isa( dist, 'nlsaPairwiseDistance' )
    if nN > getNNeighbors( dist )
        error( 'Number of nearest neighbors in the diffusion operator cannot exceed number of nearest neighbors in the pairwise distances' )
    end
    ifPrune = ~( nN == getNNeighbors( dist ) );
end
nR       = numel( partition );
epsilon  = getBandwidth( obj );
alpha    = getAlpha( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments 
Opt.batch              = 1 : getNBatch( partitionG );
Opt.logFile            = '';
Opt.logPath            = getOperatorPath( obj );
Opt.logFilePermissions = 'w';
Opt.ifWriteOperator    = true;
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
fprintf( logId, 'Number of query (out-) samples       = %i, \n', getNTotalSample( partitionO ) );
fprintf( logId, 'Number of test (in-) samples         = %i, \n', getNTotalSample( partition ) );
fprintf( logId, 'Kernel bandwidth                     = %2.4f, \n', epsilon );
fprintf( logId, 'Diffusion maps normalization (alpha) = %2.4f, \n', alpha );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the normalization and degree vectors
if alpha ~= 0
    tic
    q = getNormalization( obj ) .^ alpha;
    tWall = toc;
    fprintf( logId, 'READQ %2.4f \n', tWall );
end

tic
d = sqrt( getDegree( obj ) );
tWall = toc;
fprintf( logId, 'READD %2.4f \n', tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the global batches 
for iBG = Opt.batch

    iR  = idxG( 1, iBG );
    iB  = idxG( 2, iBG );
    nBR = getNBatch( partition( iR ) );
    iS  = getBatchLimit( partitionG, iBG );

    tic
    if alpha ~= 0
        [ pVal, pInd ] = getDistances( dist, iB, iR );
        if ifPrune
            pVal = pVal( :, 1 : nN );
            pInd = pInd( :, 1 : nN );
        end
    else
        pVal = getDistances( dist, iB, iR );
        if ifPrune
            pVal = pVal( :, 1 : nN );
        end
    end
    tWall = toc;
    fprintf( logId, 'READK %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 

    tic
    pVal = exp( - pVal / epsilon ^ 2 );
    tWall = toc;
    fprintf( logId, 'EXP %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 
        
    if alpha ~= 0
        tic
        pVal = pVal ./ q( pInd ); % q has already been raised to the power alpha
        pVal = bsxfun( @ldivide, q( iS( 1 ) : iS( 2 ) ), pVal ); 
        tWall = toc;
        fprintf( logId, 'NORMALIZEQ %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 
    end

    tic
    % d is the square root of the degree
    pVal = bsxfun( @ldivide, d( iS( 1 ) : iS( 2 ) ), pVal );
    pVal = pVal ./ d( pInd );
    tWall = toc;
    fprintf( logId, 'NORMALIZED %i/%i %i/%i %2.4f \n', iR, nR, iB, nBR, tWall ); 
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
