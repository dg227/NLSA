function [ phi, lambda, mu ] = computeEigenfunctions_par( obj, varargin )
% COMPUTEEIGENFUNCTIONS_PAR Compute diffusion eigenfunctions and Riemannian 
% measure using parallel for loops
%
% Modified 2014/06/10


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.logFile             = '';
Opt.logPath             = getOperatorPath( obj );
Opt.logFilePermissions  = 'w';
Opt = parseargs( Opt, varargin{ : } );
partition = getPartition( obj );
partitionG = mergePartitions( partition );

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
fprintf( logId, 'Number of samples            = %i, \n', getNTotalSample( obj ) );
fprintf( logId, 'Gaussian width (epsilon)     = %2.4f, \n', getEpsilon( obj ) );
fprintf( logId, 'Weight normalization (alpha) = %2.4f, \n', getAlpha( obj ) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the heat kernel
tic
nBG = getNBatch( partitionG );
pVal = cell( nBG, 1 );
pInd = cell( nBG, 1 );
for iBG = 1 : nBG
    [ pVal{ iBG }, pInd{ iBG } ] = getOperator( obj, iBG ); 
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
        y = cell( nBG, 1 );
        parfor i= 1 : nBG
            y{ i } = sum( pVal{ i } .* x( pInd{ i } ), 2 );
        end
        y = cat( 1, y{ : } );    
    end
end
