function [ c, gamma, zeta, mu ] = computeEigenfunctions( obj, diffOp, varargin )
% COMPUTEEIGENFUNCTIONS Compute eigenvalues and eigenfunctions of Koopman
% generator.
%
% diffOp is an nlsaKernelOperator object providing the basis functions.
%
% Modified 2020/04/15


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
if ~isa( diffOp, 'nlsaKernelOperator' ) || ~isscalar( diffOp )
    msgStr = [ 'Second input argument must be a scalar nlsaKernelOperator ' ...
               'object.' ];
    error( msgStr )
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse options, setup logfile and write calculation summary
Opt.ifCalcOperator                   = true;
Opt.ifWriteOperator                  = false;
Opt.ifWriteEigenfunctions            = true; 
Opt.ifWriteEigenfunctionCoefficients = true; 
Opt.logFile                          = '';
Opt.logPath                          = getOperatorPath( obj );
Opt.logFilePermissions               = 'w';

Opt = parseargs( Opt, varargin{ : } );
nEig = getNEigenfunction( obj );
idxPhi = getBasisFunctionIndices( obj );

if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), ...
                   Opt.logFilePermissions );
end
ifZeta = Opt.ifWriteEigenfunctions || nargout > 2;
clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeEigenfunctions starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of eigenfunctions       = %i \n', nEig );
fprintf( logId, 'Basis function incices         = %s \n', idx2str( idxPhi ) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read or compute the generator matrix
if ~Opt.ifCalcOperator
    tWall0 = tic;
    V = getOperator( obj ); 
    tWall = toc( tWall0 );
    fprintf( logId, 'READ %2.4f \n', tWall );
else 
    if ~isempty( Opt.logFile )
        fclose( logId );
    end
    V = computeOperator( obj, diffOp, 'logPath', Opt.logPath, ...
                                      'logFile', Opt.logFile, ...
                                      'logFilePermissions', 'a', ...
                                      'ifWriteOperator', Opt.ifWriteOperator );
    if ~isempty( Opt.logFile )
        logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'a' );
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve the generator eigenvalue problem
% c is a matrix storing the expansion coefficients of the eigenfunctions of
% the generator in the eigenbasis of diffOp
tWall0 = tic;
[ c, gamma ]  = eig( V );
gamma         = diag( gamma );
c             = c( :, 1 : nEig );
gamma         = gamma( 1 : nEig ) .'; % gamma is a row vector
tWall         = toc( tWall0 );
fprintf( logId, 'EIGV %2.4f \n', tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If requested, evaluate the eigenfunctions
if ifZeta
    tWall0 = tic;
    [ phi, mu ] = getEigenfunctions( diffOp );
    phi = phi( :, getBasisFunctionIndices( obj ) ); 
    zeta = phi * c;
    tWall = toc( tWall0 );
    fprintf( logId, 'EVALEIG %2.4f \n', tWall );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write out results
tWall0 = tic;
if Opt.ifWriteEigenfunctionCoefficients
    setEigenvalues( obj, gamma )
    setEigenfunctionCoefficients( obj, c )
end
if Opt.ifWriteEigenfunctions
    setEigenfunctions( obj, zeta, mu, '-v7.3' )
end
tWall = toc( tWall0 );
fprintf( logId, 'WRITEEIG %2.4f \n', tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clk = clock; % Exit gracefully
fprintf( logId, 'computeEigenfunctions finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end

