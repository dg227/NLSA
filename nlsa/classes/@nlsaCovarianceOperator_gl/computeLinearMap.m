function a = computeLinearMap( obj, src, varargin )
% COMPUTELINEARMAP Compute linear map from time-lagged embedded data src 
% 
% Modified 2014/07/16

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
fprintf( logId, 'computeLinearMap starting on %i/%i/%i %i:%i:%2.1f \n', ...
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
% Form linear map
a      = zeros( nDETot, nSTot );

iS1 = 1;

% Loop over the source batches
for iBG = 1 : nBG

    iR  = idxBG( 1, iBG );
    iB  = idxBG( 2, iBG );
    nB  = getNBatch( partition( iR ) );
    nSB = getBatchSize( partitionG( iBG ) );

    iS2 = iS1 + nSB - 1;

    iDE1   = 1;

    % Loop over the source components
    for iC = 1 : nC

        iDE2 = iDE1 + nDE( iC ) - 1;

        tic
        x    = getData( src( iC, iR ), iB ); 
        tWall = toc;
        fprintf( logId, 'READX component %i/%i, realization %i/%i, local batch %i/%i, global batch %i/%i %2.4f \n', ...
         iC, nC, iR, nR, iB, nB, iBG, nBG, tWall );

        tic
        a( iDE1 : iDE2, iS1 : iS2 ) = x;
        tWall = toc;
        fprintf( logId, 'ADDA component %i/%i, realization %i/%i, local batch %i/%i, global batch %i/%i %2.4f \n', ...

        iDE1 = iDE2 + 1;
    end % component loop 
    iS1 = iS2 + 1;
end % batch loop

if Opt.ifWriteOperator
    tic
    setLinearMap( obj, a, '-v7.3' )
    tWall = toc;
    fprintf( logId, 'WRITEA %2.4f \n', tWall )
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeLinearMap finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end

