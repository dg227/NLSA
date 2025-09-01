function a = computeLinearMap( obj, prj, varargin )
% COMPUTELINEARMAP Compute linear maps from projected embedded data prj
% 
% Modified 2015/10/19

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do some basic checking of input arguments
if ~isscalar( obj )
    error( 'First argument must be scalar' )
end
if ~isa( prj, 'nlsaProjectedComponent' )
    error( 'Target data must be specified as anarray of nlsaProjected objects' )
end
if ~isCompatible( obj, prj )
    error( 'Incompatible embedded data' )
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup a global partition for the embedded data
% idxBG are the global indices of the batch boundaries
partition = getPartition( obj );
[ partitionG, idxBG ] = mergePartitions( partition );
partitionD = getSpatialPartition( obj );
nDE    = getBatchSize( partitionD );
nC     = numel( nDE );
nR     = numel( partition );
nSB    = getBatchSize( partitionG );
nSTot  = getNSample( partitionG );
nBG    = getNBatch( partitionG );
nDETot = sum( nDE );
nL     = getNEigenfunction( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.logFile           = '';
Opt.logPath           = getOperatorPath( obj );
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
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeLinearMap starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of samples                       = %i \n', nSTot );
fprintf( logId, 'Number of realizations                  = %i \n', nR );
fprintf( logId, 'Number of components                    = %i \n', nC );
fprintf( logId, 'Number of batches                       = %i \n', nBG );
fprintf( logId, 'Max batch size                          = %i \n', max( nSB ) );
fprintf( logId, 'Min batch size                          = %i \n', min( nSB ) );
fprintf( logId, 'Max compoment embedding space dimension = %i \n', max( nDE ) );
fprintf( logId, 'Min component embedding space dimension = %i \n', min( nDE ) );
fprintf( logId, 'Temporal space dimension                = %i \n', nL );
fprintf( logId, '----------------------------------------- \n' ); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Form linear map
tic
a = getProjectedData( prj, 1 : nC, getBasisFunctionIndices( obj ) );
tWall = toc;
fprintf( logId, 'READA %2.4f \n' );


if Opt.ifWriteOperator
    tic
    setLinearMap( obj, a, '-v7.3' )
    tWall = toc;
    fprintf( logId, 'WRITEA %2.4f \n', tWall );
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeLinearMap finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end

