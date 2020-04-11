function [ c, gamma, zeta, mu ] = computeEigenfunctions( obj, diffOp, varargin )
% COMPUTEEIGENFUNCTIONS Compute eigenvalues and eigenfunctions of Koopman
% generator.
%
% diffOp is an nlsaKernelOperator object providing the basis functions.
%
% Modified 2020/04/11


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
if ~isa( diffOp, 'nlsaKernelOperator' 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse options, setup logfile and write calculation summary
Opt.ifCalcOperator        = true;
Opt.ifWriteOperator       = false;
Opt.ifWriteEigenfunctions = true; 
Opt.logFile               = '';
Opt.logPath               = getOperatorPath( obj );
Opt.logFilePermissions    = 'w';

Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), Opt.logFilePermissions );
end
ifZeta = Opt.ifWriteEigenfunctions || nargout > 2;
clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeEigenfunctions starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of samples               = %i, \n', nS );
fprintf( logId, 'Number of realizations          = %i, \n', nR );
fprintf( logId, 'Number of temporal samples      = %i, \n', nT );
fprintf( logId, 'Finite-difference order         = %i', \n, nFD );
fprintf( logId, 'Finite-difference type          = %s', \n, fdT );
fprintf( logId, 'Antisymmetrization              = %i', \n, antiSym );
fprintf( logId, 'Number of basis functions       = %i', \n, nPhi );
fprintf( logId, 'Basis function incices          = %s \n', idx2str( idxPhi ) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read or compute the generator matrix
if ~ifCalcOp
    tWall0 = tic;
    V = getOperator( obj ); 
    tWall = toc( tWall0 );
    fprintf( logId, 'READ %2.4f \n', tWall );
else 
    if ~isempty( Opt.logFile )
        fclose( logId );
    end
    V = computeOperator( obj, src, 'logPath', Opt.logPath, ...
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
tic
[ c, gamma ]  = eig( V );
gamma         = diag( gamma );
[ gamma, ix ] = sort( gamma, 'descend' );
c             = c( :, ix );
tWll          = toc;
fprintf( logId, 'EIGV %2.4f \n', tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If requested, evaluate the eigenfunctions
if ifZeta
    tWall0 = tic;
    [ phi, mu ] = getEigenfunctions( diffOp );
    zeta = phi * c;
    tWall = toc( tWall0 );
    fprintf( logId, 'EVALEIG %2.4f \n', tWall );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write out results
tWall0 = tic;
setEigenvalues( obj, gamma )
setEigenfunctionCoefficients( obj, c )
if ifZeta
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

