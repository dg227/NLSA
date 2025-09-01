function c = computeLeftCovariance( obj, src, varargin )
% COMPUTELEFTCOVARIANCE Compute left (spatial) covariance matrix from 
% time-lagged embedded data src 
% 
% Modified 2014/07/16


uuuu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do some basic checking of input arguments
if ~isscalar( obj )
    error( 'First argument must be scalar' )
end
if ~isa( src, 'nlsaEmbeddedComponent' )
    error( 'Embedded data must be specified as an array of nlsaEmbeddedComponent objects' )
end
if ~iscompatible( obj, src )
    error( 'Incompatible embedded data' )
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup a global partition for the embedded data
% idxBG are the global indices of the batch boundaries
partition = getPartition( obj );
[ partitionG, idxBG ] = mergePartitions( partition );
partitionD = getSpatialPartition( obj );
nDE    = getBatchSize( partitionD );
nC     = numel( nD );
nR     = numel( partition );
nSB    = getBatchSize( partitionG );
nSTot  = getNSample( partitionG );
nBG    = getNBatch( partitionG );
nDETot = sum( nDE );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.logFile           = '';
Opt.logPath           = obj.path;
Opt.logFilePermission = 'w';
Opt.ifWriteOperator   = true;
Opt = parseargs( Opt, varargin{ : } );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), Opt.logFilePermission );
end

clk = clock;
[ status hostname ] = unix( 'hostname' );
fprintf( logId, 'computeLeftCovariance starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Total dimension                         = %i \n', sum( nDETot ) );
fprintf( logId, 'Number of samples                       = %i \n', nSTot );
fprintf( logId, 'Number of realizations                  = %i \n', nR );
fprintf( logId, 'Number of components                    = %i \n', nC );
fprintf( logId, 'Number of batches                       = %i \n', nB );
fprintf( logId, 'Max batch size                          = %i \n', max( nSB ) );
fprintf( logId, 'Min batch size                          = %i \n', min( nSB ) );
fprintf( logId, 'Max compoment embedding space dimension = %i \n', max( nD ) );
fprintf( logId, 'Min component embedding space dimension = %i \n', min( nD ) );
fprintf( logId, '----------------------------------------- \n' ); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Form left covariance
cu = zeros( nDETot );

% Loop over the source batches
for iBG = 1 : nBG

    iR  = idxBG( 1, iBG );
    iB  = idxBG( 2, iBG );
    nB  = getNBatch( partition( iR ) );
    nSB = getBatchSize( partitionG( iBG ) );

    iD1 = 1;

    % Loop over the components
    for iC = 1 : nC

        iD2 = iD1 + nD( iC ) -1;

        tic
        xI    = getData( src( iC, iR ), iB ); 
        tWall = toc;
        fprintf( logId, 'READXI Global batch %i/%i, component %i/%i %2.4f \n', ...
            iBG, nBG, iC, nC, tWall );

        tic
        cu( iD1 : iD2, iD1 : iD2 ) = cu( iD1 : iD2, iD1 : iD2 ) + xI * xI';
        tWall = toc;
        fprintf( logId, 'CU Global batch %i/%i, components %i-%i %2.4f \n', ...
            iBG, nBG, iC, iC, tWall );

        jD1 = iD2 + 1;

        for jC = iC + 1 : nC

 
            jD2 = jD1 + nD( jC ) - 1;

            tic
            xJ = getData( src( jC, iR ), iB );
            tWall = toc;
            fprintf( logId, 'READXJ Global batch %i/%i, component %i/%i %2.4f \n', ...
                iBG, nBG, jC, nC, tWall );

            tic
            cu( iD1 : iD2, jD1 : jD2 ) = c( iD1 : iD2, jD1 : jD2 ) + xI' * xJ;
            tWall = toc;
            fprintf( logId, 'CU Global batch %i/%i, components %i-%i %2.4f \n', ...
                iBG, nBG, iC, jC, tWall );

            jD1 = jD2 + 1;
        end % J component loop

        iD1 = iD2 + 1;

    end % I component loop
end % batch loop 

% Fill in the symmetric blocks
iD1 = 1;
for iC = 1 : nC
    iD2 = iD1 + nD( iC ) - 1;
    tic 
    cu( iS1 + 1 : end, iS1 : iS2 ) = cu( iS1 : iS2, iS1 + 1 : end )';
    tWall = toc;
    fprintf( logId, 'SYM component %i %2.4f \n', iC, tWall );
end

if Opt.ifWriteOperator
    tic
    setLeftCovariance( obj, cu, '-v7.3' )
    tWall = toc;
    fprintf( logId, 'WRITECU %2.4f \n', tWall )
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeLeftCovariance finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
