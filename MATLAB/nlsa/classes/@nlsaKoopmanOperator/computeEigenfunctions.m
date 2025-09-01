function [ c, gamma, zeta, mu, cL, zetaL ] = computeEigenfunctions( obj, ...
                                                 diffOp, varargin )
% COMPUTEEIGENFUNCTIONS Compute eigenvalues, eigenfunctions, and left 
% eigenfunctions of Koopman generator.
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
% zeta: A matrix of size [ nS nPhi ] storing the values of the Koopman 
%      eigenfunctions. nS is the number of samples. 
%
% mu: A column vector of size [ nS 1 ] storing the inner product weights with
%     repect to which the phi are orthonormal. 
%
% cL: A matrix of size [ nPhi nEig ] storing the expansion coefficients of the
%     left eigenfunctions. 
%
% zetaL: A matrix of size [ nS nPhi ] storing the values of the left 
%        eigenfunctions. 
%
% Modified 2020/08/28


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
Opt.ifLeftEigenfunctions             = false;
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
ifZeta  = Opt.ifWriteEigenfunctions || nargout > 2;
ifCL    = Opt.ifLeftEigenfunctions || nargout > 4;
ifZetaL = ( ifCL && Opt.ifWriteEigenfunctions ) || nargout == 5;  
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
% If requested compute the left eigenvectors. 
% cL is a matrix storing the expansion coefficients of the left eigenfunctions 
% the generator in the eigenbasis of diffOp
if ifCL
    tWall0 = tic;
    S = c * c';  % analysis operator
    cL = S \ c;  % dual basis
    tWall         = toc( tWall0 );
    fprintf( logId, 'EIGVL %2.4f \n', tWall );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Truncate to the selected number of eigenfunctions
c = c( :, 1 : nEig );
gamma = gamma( 1 : nEig ) .'; % gamma is a row vector
if ifCL
    cL  = cL( :, 1 : nEig );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If requested, evaluate the eigenfunctions
if ifZeta
    tWall0 = tic;
    [ phi, mu ] = getEigenfunctions( diffOp );
    phi = phi( :, idxPhi ); 
    zeta = phi * c;
    tWall = toc( tWall0 );
    fprintf( logId, 'EVALEIG %2.4f \n', tWall );
end
if ifZetaL
    tWall0 = tic;
    zetaL = phi * cL;
    tWall = toc( tWall0 );
    fprintf( logId, 'EVALEIGL %2.4f \n', tWall );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write out results
tWall0 = tic;
if Opt.ifWriteEigenfunctionCoefficients
    setEigenvalues( obj, gamma )
    setEigenfunctionCoefficients( obj, c )
    if Opt.ifLeftEigenfunctions
        setLeftEigenfunctionCoefficients( obj, cL )
    end
end
if Opt.ifWriteEigenfunctions
    setEigenfunctions( obj, zeta, mu, '-v7.3' )
    if Opt.ifLeftEigenfunctions
        setLeftEigenfunctions( obj, zetaL, mu, '-v7.3' )
    end
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

