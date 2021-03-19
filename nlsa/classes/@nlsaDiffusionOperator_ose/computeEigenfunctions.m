function phi = computeEigenfunctions( obj, diffOp, varargin )
% COMPUTEEIGENFUNCTIONS Perform out-of-sample extension of the 
% eigenfunctions of a diffusion operator diffOp
% 
% Modified 2020/01/30

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
if ~isa( diffOp, 'nlsaDiffusionOperator' )
    error( 'Second argument must be an nlsaDiffusionOperator object' )
end
partition  = getPartitionTest( obj );
if getNTotalSample( partition ) ~= getNTotalSample( getPartition( diffOp ) ) 
    error( 'Incompatible number of test samples' )
end
partitionO = getPartition( obj );
nPhiO      = getNEigenfunction( obj );
nPhi       = getNEigenfunction( diffOp );
if nPhiO > nPhi
    error( 'Number of OSE eigenfunctions cannot exceed the number of in-sample eigenfunctions' )
end
[ partitionG, idxG ] = mergePartitions( partitionO ); % global partition
nR  = numel( partitionO );
nST = getNTotalSample( partition );
beta = getBeta( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
Opt.batch                 = 1 : getNBatch( partitionG );
Opt.logFile               = '';
Opt.logPath               = obj.path;
Opt.logFilePermissions    = 'w';

Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), ...
                    Opt.logFilePermissions );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeEigenfunctions starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of test (in-) samples = %i, \n', getNTotalSample( partition ) );
fprintf( logId, 'Number of query (out-) samples = %i, \n', getNTotalSample( partitionO ) );
fprintf( logId, 'Number of test eigenfunctions = %2.4f, \n', nPhi );
fprintf( logId, 'Number of OSE eigenfunctions = %2.4f, \n', nPhiO );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
phi    = getEigenfunctions( diffOp, [], [], [], 'ifMu', false );
phi    = phi( :, 1 : nPhiO ); 
lambda = getEigenvalues( diffOp );
lambda = lambda( 1 : nPhiO );
tWall = toc;
fprintf( logId, 'READPHI  %i %i  %2.4f \n', nST, nPhi, tWall ); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iBG = Opt.batch

    iR = idxG( 1, iBG );
    iB = idxG( 2, iBG );
    nB = getNBatch( partitionO( iR ) );
    nSB = getBatchSize( partitionO( iR ), iB );

    tic
    [ pVal, pInd ] = getOperator( obj, iB, iR );
    tWall = toc;
    fprintf( logId, 'READP %i/%i %i/%i %2.4f \n', iR, nR, iB, nB, tWall );

    tic
    phiO = zeros( nSB, nPhiO );
    for iPhi = 1 : nPhiO 
        phiVals = phi( :, iPhi ).';
        phiO( :, iPhi ) = sum( pVal .* phiVals( pInd ), 2 );  
    end
    phiO  = bsxfun( @rdivide, phiO, lambda' );
    if beta ~= 0
        muO = phiO( :, 1 ) .^ 2;
    else
        nSO = size( phiO, 1 ); 
        muO = ones( nSO, 1 ) / nST;
    end

    tWall = toc;
    fprintf( logId, 'PHIOSE %i/%i %i/%i %2.4f \n', iR, nR, iB, nB, tWall );

    tic
    setEigenfunctions( obj, phiO, muO, iB, iR, '-v7.3' )
    tWall = toc;
    fprintf( logId, 'WRITEPHI %i/%i %i/%i %2.4f \n', iR, nR, iB, nB, tWall );
end


clk = clock; % Exit gracefully
fprintf( logId, 'computeEigenfunctions finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
