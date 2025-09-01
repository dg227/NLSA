function [ v, lambda ] = computeRightEigenfunctions( obj, varargin )
% COMPUTERIGHTEIGENFUNCTIONS Compute right (temporal) covariance eigenfunctions 
%
% Modified 2016/02/02

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
if isa( varargin{ 1 }, 'nlsaEmbeddedComponent' )
    ifCalcC = true;
    src = varargin{ 1 };
    varargin = varargin( 2 : end );
else
    ifCalcC = false;
end

Opt.logFile             = '';
Opt.logPath             = getOperatorPath( obj );
Opt.logFilePermissions  = 'w';
Opt.ifWriteOperator     = false;

Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), Opt.logFilePermissions );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeRightEigenfunctions starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of samples            = %i, \n', getNTotalSample( obj ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read or compute the  kernel
if ~ifCalcC
    tic
    c = getRightCovariance( obj ); 
    tWall = toc;
    fprintf( logId, 'READ %2.4f \n', tWall );
else 
    if ~isempty( Opt.logFile )
        fclose( logId );
    end
    c = computeRightCovariance( obj, src, 'logPath', Opt.logPath, ...
                                     'logFile', Opt.logFile, ...
                                     'logFilePermissions', 'a', ...
                                     'ifWriteOperator', false );
    if ~isempty( Opt.logFile )
        logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'a' );
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve the eigenvalue problem
tic
opts.disp      = 1;
[ v, lambda ]  = eigs( c, getNEigenfunction( obj ), 'lm', opts );
lambda         = diag( lambda );
[ lambda, ix ] = sort( lambda, 'descend' );
v              = v( 1 : end, ix );
tWall          = toc;
fprintf( logId, 'EIGC %2.4f \n', tWall );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write out results
tic
setRightSingularVectors( obj, v )
setSingularValues( obj, sqrt( lambda ) )
tWall = toc;
fprintf( logId, 'WRITEV %2.4f \n', tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clk = clock; % Exit gracefully
fprintf( logId, 'computeRightEigenfunctions finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end

