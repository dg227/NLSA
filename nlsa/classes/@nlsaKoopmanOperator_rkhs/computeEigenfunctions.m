function [ c, gamma, E, zeta, mu ] = computeEigenfunctions( ...
    obj, diffOp, varargin )
% COMPUTEEIGENFUNCTIONS Compute eigenvalues and eigenfunctions of compactified
% Koopman generator.
%
% diffOp is an nlsaKernelOperator object providing the basis functions.
%
% Output arguments:
% 
% c: A matrix of size [ nPhi nEig ] storing the expansion coefficients of the
%    Koopman eigenfunctions. nPhi is the number of diffusion eigenfunctions
%    employed, and nEig is the number of computed Koopman eigenfunctions.
%
% gamma: A row vector of size [ 1 nEig ] storing the generator eigenvalues.
%
% E: A row vector of size [ 1 nEig ] storing the Dirichlet energies of the
%    eigenfunctions.
%
% zeta: A matrix of size [ nS nPhi ] storing the values of the kernel
%      eigenfunctions employed. nS is the number of samples. 
%
% mu: A column vector of size [ nS 1 ] storing the inner product weights with
%     repect to which the phi are orthonormal. 
%
% Modified 2020/05/01


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
gamma         = diag( gamma ).';
tWall         = toc( tWall0 );
fprintf( logId, 'EIGV %2.4f \n', tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tWall0 = tic;

% Compute Dirichlet energies of eigenvectors
Lambda = getEigenvalues( diffOp );
Lambda = Lambda( idxPhi );
Lambda = Lambda( : );
eta = computeRegularizingEigenvalues( obj, diffOp );
eta = eta( : );  % ensure eta is a column vector
epsilon = getRegularizationParameter( obj );
dt = getSamplingInterval( obj );
lambda = exp( - epsilon * eta );
omega = imag( gamma )';
%E = sum( lambda .* abs( c ) .^ 2, 1 );  
%E = ( 1 ./ E - 1 ) ./ ( 1 - ( imag( gamma ) * dt ) .^ 2 );
%E = ( 1 ./ E - 1 );
l2Norm = sum( lambda .* abs( c ) .^ 2, 1 );
hNorm = sum( lambda ./ Lambda .* abs( c ) .^2, 1 );
E = hNorm ./ l2Norm - 1; 

% Sort results in order of increasing Dirichlet energy
[ E, idxE ] = sort( E, 'ascend' );
E = E( 1 : nEig );
idxE = idxE( 1 : nEig );
gamma = gamma( idxE );
c = c( :, idxE );
tWall = toc( tWall0 );
fprintf( logId, 'ENGY %2.4f \n', tWall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If requested, evaluate the eigenfunctions
if ifZeta
    tWall0 = tic;
    [ phi, mu ] = getEigenfunctions( diffOp );
    phi = phi( :, getBasisFunctionIndices( obj ) ); 
    sqrtLambda = exp( - epsilon * eta / 2 )';
    zeta = phi * c .* sqrtLambda;
    tWall = toc( tWall0 );
    fprintf( logId, 'EVALEIG %2.4f \n', tWall );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tWall0 = tic;
setEigenvalues( obj, gamma, E )
if Opt.ifWriteEigenfunctionCoefficients
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

