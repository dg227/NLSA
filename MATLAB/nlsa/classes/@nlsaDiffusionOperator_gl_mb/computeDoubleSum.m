function dSum = computeDoubleSum( obj, dist, varargin )
% COMPUTEDOUBLESUM Compute double kernel sum from candidate bandwidths
% 
% Modified 2015/05/08

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
if ~isa( dist, 'nlsaSymmetricDistance_gl' )
    error( 'Distance data must be specified as an nlsaSymmetricDistance_gl object' )
end
partition = getPartition( obj );
if any( ~isequal( partition, getPartition( dist ) ) )
    error( 'Incompatible distance data partition' )
end
nS      = getNTotalSample( partition );
epsilon = getBandwidths( obj );
nE      = numel( epsilon );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
Opt.logFile            = '';
Opt.logPath            = getOperatorPath( obj );
Opt.logFilePermissions = 'w';
Opt.ifWrite            = true;
Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), ...
                   Opt.logFilePermissions );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeDoubleSum starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Number of samples            = %i, \n', nS );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read input data yRow yCol yVal
tic
[ ~, ~, yVal ] = getDistances( dist );
tWall = toc;
fprintf( logId, 'READDIST %2.4f \n', tWall );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the bandwidths
dSum = zeros( 1, nE );
for iE = 1 : nE
    tic
    dSum( iE ) = sum( exp( -yVal / epsilon( iE ) ^ 2 ) );
    tWall = toc;
    fprintf( logId, 'SUM, %i/%i %2.4f, \n', iE, nE );
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write out results
tic
if Opt.ifWrite
    tic
    setDoubleSum( obj, dSum, '-v7.3' )
    fprintf( logId, 'WRITE %2.4f, \n',  tWall );
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeDoubleSum finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
