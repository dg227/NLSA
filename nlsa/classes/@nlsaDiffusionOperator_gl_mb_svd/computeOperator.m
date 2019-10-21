function [ l, q, d ] = computeOperator( obj, dist, varargin )
% COMPUTETEOPERATOR Compute heat kernel from symmetrized distance data dist
% 
% Modified 2018/06/10

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
epsilon = getBandwidth( obj );
alpha   = getAlpha( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
Opt.logFile            = '';
Opt.logPath            = getOperatorPath( obj );
Opt.logFilePermissions = 'w';
Opt.ifWriteOperator    = true;
Opt = parseargs( Opt, varargin{ : } );
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
fprintf( logId, 'Number of samples            = %i, \n', nS );
fprintf( logId, 'Gaussian width (epsilon)     = %2.4f, \n', epsilon );
fprintf( logId, 'Weight normalization (alpha) = %2.4f, \n', alpha );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read input data yRow yCol yVal
tic
[ yRow, yCol, yVal ] = getDistances( dist );
yRow = double( yRow );
yCol = double( yCol );
nNZ = numel( yVal );
tWall = toc;
fprintf( logId, 'READDIST %i samples, %i matrix elements %2.4f \n', ...
         nS, nNZ, tWall );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the unnormalized weight matrix
tic
yVal = exp( -yVal / epsilon ^ 2 ); % yVal is distance ^ 2
l = sparse( yRow, yCol, yVal, nS, nS, nNZ );
tWall = toc;
fprintf( logId, 'WMAT, %i nonzero entries, %2.4f avg. per row %2.4f, \n', ...
         nNZ, nNZ / nS, tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If required, apply non-isotropic normalization
if alpha ~= 0
    tic
    if alpha ~= 1
        q = full( sum( l, 1 ) ) .^ alpha;
    else
        q = full( sum( l, 1 ) ); % don't exponentiate if alpha == 1
    end
    q = q';
    yVal = yVal ./ q( yRow ) ./ q( yCol );
    l = sparse( yRow,  yCol, yVal, nS, nS, nNZ );
    tWall = toc;
    fprintf( logId, 'NORMALIZATION (alpha) = %2.4f  %2.4f, \n', alpha, tWall );
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Form the normalized Laplacian
tic
d = full( sum( l, 1 ) );
d = d';
if alpha == 0
    q = d;
end
yVal = yVal ./ d( yRow );
l = sparse( yRow, yCol, yVal, nS, nS, nNZ );
tWall = toc;
fprintf( logId, 'NORMALIZATION (Markov) %2.4f, \n',  tWall );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write out results
tic
if alpha == 0
    q = d;
else
    q = q .^ ( 1 / alpha );
end
if Opt.ifWriteOperator
    setOperator( obj, l, d, q, '-v7.3' )
else
    setOperator( obj, [], d, q, '-v7.3' )
end
tWall = toc;
fprintf( logId, 'WRITE %2.4f, \n',  tWall );

clk = clock; % Exit gracefully
fprintf( logId, 'computeOperator finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
