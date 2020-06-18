function a = computeProjection( obj, src, kOp, varargin )
% COMPUTEPROJECTION Compute projected data from time-lagged embedded data src
%  and kernel operator kOp 
% 
% Modified 2020/06/16

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Validate input arguments
if ~iscolumn( obj )
    error( 'First argument must be a column vector' )
end
if ~isa( src, 'nlsaEmbeddedComponent' )
    error( 'Source data must be specified as an array of nlsaEmbeddedComponent objects' )
end
if ~isa( kOp, 'nlsaKernelOperator' ) || ~isscalar( kOp )
    error( 'Kernel operator must be specified as a scalar nlsaKernelOperator object' )
end
if ~isCompatible( obj, src )
    error( 'Incompatible source components' )
end
if ~isCompatible( obj, kOp )
    error( 'Incompatible kernel operator' )
end
nDE = getEmbeddingSpaceDimension( obj );
nL  = getNBasisFunction( obj );
[ nC, nR ] = size( src );
[ partition, idxG ] = mergePartitions( getPartition( obj ) );
nB     = getNBatch( partition );
nSB    = getBatchSize( partition );
nS     = getNTotalSample( partition );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
Opt.component         = 1 : nC;
Opt.logFile           = '';
Opt.logPath           = obj( 1 ).path;
Opt.logFilePermission = 'w';
Opt.ifWriteA          = true;
Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), Opt.logFilePermission );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeProjection starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of samples                       = %i \n', nS );
fprintf( logId, 'Number of realizations                  = %i \n', nR );
fprintf( logId, 'Number of components                    = %i \n', nC );
fprintf( logId, 'Number of batches                       = %i \n', nB );
fprintf( logId, 'Max batch size                          = %i \n', max( nSB ) );
fprintf( logId, 'Min batch size                          = %i \n', min( nSB ) );
fprintf( logId, 'Max compoment embedding space dimension = %i \n', max( nDE ) );
fprintf( logId, 'Min component embedding space dimension = %i \n', min( nDE ) ); 
fprintf( logId, 'Max number of basis functions           = %i \n', max( nL ) );
fprintf( logId, 'Min number of basis functions           = %i \n', min( nL ) );
fprintf( logId, '----------------------------------------- \n' ); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read eigenfunctions.
% Complex conjugation is performed since we will be doing L2 projection.
tic
[ phi, mu ] = getEigenfunctions( kOp );
phi         = phi .* mu;
phi         = conj( phi ); 
tWall = toc;
fprintf( logId, 'READPHI number of samples %i, number of eigenfunctions %i, %2.4f \n', ...
             size( phi, 1 ), size( phi, 2 ), tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the components
for iC = Opt.component

    % Allocate projected data array
    tic
    a = zeros( nDE( iC ), nL( iC ) );
    tWall = toc;
    fprintf( logId, 'ALLOCA, component %i, %i %2.4f \n', iC, nC, tWall );

    % Loop over the source batches
    iS1 = 1;

    for iB = 1 : nB
        iR     = idxG( 1, iB );
        iBR    = idxG( 2, iB );
        nBR    = getNBatch( src( iC, iR ) );
        iS2    = iS1 + nSB( iB ) - 1;

        % Read batch data
        tic
        x    = getData( src( iC, iR ), iBR ); 
        tWall = toc;
        fprintf( logId, 'READX, component %i/%i, realization %i/%i, local batch %i/%i, global batch %i/%i %2.4f \n', ...
             iC, nC, iR, nR, iBR, nBR, iB, nB, tWall );

        % Perform projection
        tic

        a = a + x * phi( iS1 : iS2, 1 : nL( iC ) );
        tWall = toc;
        fprintf( logId, 'PROJ %i samples, embedding space dimension %i %2.4f\n', ...
                nSB, nDE, tWall );  
        iS1 = iS2 + 1;
    end % batch loop


    if Opt.ifWriteA
        tic
        setProjectedData( obj( iC ), a, '-v7.3' );
        tWall = toc;
        fprintf( logId, 'WRITEA, component %i, %i %2.4f \n', iC, nC, tWall );
    end
    
end % component loop

clk = clock; % Exit gracefully
fprintf( logId, 'computeProjection finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
