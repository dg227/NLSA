function [ phi, lambda, mu ] = computeEigenfunctions( obj, varargin )
% COMPUTEEIGENFUNCTIONS Compute diffusion eigenfunctions and Riemannian 
% measure 
%
% Modified 2018/06/18


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
if isa( varargin{ 1 }, 'nlsaSymmetricDistance_gl' )
    ifCalcP = true;
    src = varargin{ 1 };
    varargin = varargin( 2 : end );
else
    ifCalcP = false;
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
fprintf( logId, 'computeEigenfunctions starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of samples            = %i, \n', getNTotalSample( obj ) );
fprintf( logId, 'Gaussian width (epsilon)     = %2.4f, \n', getBandwidth( obj ) );
fprintf( logId, 'Weight normalization (alpha) = %2.4f, \n', getAlpha( obj ) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read or compute the heat kernel
if ~ifCalcP
    tic
    p = getOperator( obj ); 
    tWall = toc;
    fprintf( logId, 'READ %2.4f \n', tWall );
else 
    if ~isempty( Opt.logFile )
        fclose( logId );
    end
    p = computeOperator( obj, src, 'logPath', Opt.logPath, ...
                                   'logFile', Opt.logFile, ...
                                   'logFilePermissions', 'a', ...
                                   'ifWriteOperator', Opt.ifWriteOperator );
    if ~isempty( Opt.logFile )
        logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'a' );
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve the sparse eigenvalue problem
tic
opts.disp      = 1;
[ v, lambda ]  = eigs( p, getNEigenfunction( obj ), 'lm', opts );
lambda         = diag( lambda );
[ lambda, ix ] = sort( lambda, 'descend' );
v              = v( 1 : end, ix );
mu             = v( :, 1 ) .^ 2;
if v( 1, 1 ) < 0
    v( :, 1 ) = -v( :, 1 );
end
tWall          = toc;
fprintf( logId, 'EIGP %2.4f \n', tWall );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write out results
tic
setEigenfunctions( obj, v, mu, '-v7.3' )
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

