function [ phi, lambda, mu ] = computeEigenfunctions_spmd( obj, varargin )
% COMPUTEEIGENFUNCTIONS_PAR Compute diffusion eigenfunctions and Riemannian 
% measure using SPMD parallelilzation 
%
% Modified 2014/06/30


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.logFile            = '';
Opt.logPath            = getOperatorPath( obj );
Opt.logFilePermissions = 'w';
Opt.idxType            = 'double';
Opt = parseargs( Opt, varargin{ : } );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get object properties needed in the eigenvalue problem
partition  = getPartition( obj );
partitionG = mergePartitions( partition );
precV      = getPrecisionEigs( obj );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup log file, write calculation summary
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), Opt.logFilePermissions );
end
clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeEigenfunctions starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of samples            = %i \n', getNTotalSample( obj ) );
fprintf( logId, 'Gaussian width (epsilon)     = %2.4f \n', getEpsilon( obj ) );
fprintf( logId, 'Weight normalization (alpha) = %2.4f \n', getAlpha( obj ) );
fprintf( logId, 'Precision                    = %s \n', precV );
fprintf( logId, 'Array index type             = %s \n', Opt.idxType );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the heat kernel
tic
nB = getNBatch( partitionG );  % total number of batches 
nN = getNNeighbors( obj ); 
spmd
    codistr = codistributor1d( 1, ...
                               codistributor1d.unsetPartition, ...
                               [ nB 1 ] );

    [ idxBL( 1 ), idxBL( 2 ) ] = codistr.globalIndices( 1, labindex ); % batch indices in lab
    idxSLim = getBatchLimit( partitionG, idxBL );
    idxSLim = [ idxSLim( 1 ) idxSLim( end ) ]; % sample indices in lab
    nSL     = idxSLim( 2 ) - idxSLim( 1 ) + 1;     % sample number in lab
    [ pVal, pInd ] = getOperator( obj, idxBL( 1 ) : idxBL( 2 ), [], precV, Opt.idxType );
    yL  = zeros( nSL, 1, precV );
    xL  = zeros( nSL, nN, precV );
    pXL = zeros( nSL, nN, precV );
end
tWall = toc;
fprintf( logId, 'READ %2.4f \n', tWall );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve the sparse eigenvalue problem
tic
opts.disp      = 1;
[ v, lambda ]  = eigs( @pEval, getNTotalSample( partition ), ...
                               getNEigenfunction( obj ) + 1, 'lm', opts );
lambda         = diag( lambda );
[ lambda, ix ] = sort( lambda, 'descend' );
v              = v( 1 : end, ix );
mu             = v( :, 1 ) .^ 2;
tWall          = toc;
fprintf( logId, 'EIGP %2.4f \n', tWall );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write out results
tic
iS1 = 1;
for iR = 1 : numel( partition )
    for iB = 1 : getNBatch( partition( iR ) );
        iS2 = iS1 + getBatchSize( partition( iR ), iB ) - 1;
            setEigenfunctions( obj, v( iS1 : iS2, : ), mu( iS1 : iS2 ), ...
                               iB, iR, '-v7.3' )
        iS1 = iS2 + 1;
    end
end
setEigenvalues( obj, lambda )
tWall = toc;
fprintf( logId, 'WRITEPHI %2.4f \n', tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clk = clock; % Exit gracefully
fprintf( logId, 'computeEigenfunctions finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Multiplication by the heat kernel
    function y = pEval( x )
        tic
        if strcmp( precV, 'single' )
            x = single( x );
        end
        spmd
            xL( :, : )  = x( pInd );
            pXL( :, : ) = pVal .* xL; 
            yL( : )     = sum( pXL, 2 );
        end
        y = double( cat( 1, yL{ : } ) );
        toc
    end
end
