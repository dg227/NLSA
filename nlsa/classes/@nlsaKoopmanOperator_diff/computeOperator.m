function V = computeOperator( obj, diffOp, varargin )
% COMPUTEOPERATOR Compute Koopman generator in an eigenbasis of a kernel
% integral operator.
% 
% Modified 2020/04/12

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.logFile            = '';
Opt.logPath            = obj.path;
Opt.logFilePermissions = 'w';
Opt.ifWriteOperator    = true;
Opt = parseargs( Opt, varargin{ : } );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the raw generator using the parent method.
% Input arguments are validated using the parent method.

V = computeOperator@nlsaKoopmanOperator( obj, diffOp, ...
        'logPath',            Opt.logPath, ...
        'logFile',            Opt.logFile, ...
        'logFilePermissions', Opt.logFilePermissions;
        'ifWriteOperator',    false );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'a' );
end

regType = getRegularizationType( obj );
epsilon = getRegularizationParameter( obj );
clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeOperator starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Regularization type           = %s, \n', regType );
fprintf( logId, 'Regularization parameter      = %2.3g, \n', epsilon );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tWall0 = tic;
% Compute the regularizing eigenvalues
eta = computeRegularizingEigenvalues( obj, diffOp );

% Form regularized operator
epsilon = getRegularizationParameter( obj );
V = V - epsilon * diag( eta );  

tWall = toc( tWall0 );
fprintf( logId, 'REGV %2.4f \n', tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Opt.ifWriteOperator
    tWall0 = tic;
    setOperator( obj, V, '-v7.3' )
    tWall = toc( tWall0 );
    fprintf( logId, 'WRITEV %2.4f \n', tWall ); 
    end
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeOperator finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
