function [ c, gamma, eta, zeta, mu ] = computeEigenfunctions( ...
    obj, diffOp, varargin )
% COMPUTEEIGENFUNCTIONS Compute eigenvalues and eigenfunctions of regularized
% Koopman generator.
%
% diffOp is an nlsaKernelOperator object providing the basis functions.
%
% Modified 2020/04/15

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.ifCalcOperator                   = true;
Opt.ifWriteOperator                  = false;
Opt.ifWriteEigenfunctions            = true; 
Opt.ifWriteEigenfunctionCoefficients = true; 
Opt.logFile                          = '';
Opt.logPath                          = getOperatorPath( obj );
Opt.logFilePermissions               = 'w';
Opt = parseargs( Opt, varargin{ : } );

% Open logfile 
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), ...
                   Opt.logFilePermissions );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute eigenfunctions using the parent method. 
% Parent method validates input arguments.
% We set all write options for the eigenfunctions/coefficients to false as
% we do writing from this method. 

ifZeta = nargout > 3 || Opt.ifWriteEigenfunctions;

parentOpt = { 'ifCalcOperator',                   Opt.ifCalcOperator, ...
              'ifWriteOperator',                  Opt.ifWriteOperator, ...
              'ifWriteEigenfunctions',            false, ...
              'ifWriteEigenfunctionCoefficients', false, ...
              'ifWriteEigenfunctions',            false,  ...
              'logPath',                          Opt.logPath, ...
              'logFile',                          Opt.logFile, ...
              'logFilePermissions',               'a' };
               
if ifZeta
    [ c, gamma, zeta, mu ] = computeEigenfunctions@nlsaKoopmanOperator( ...
        obj, diffOp, parentOpt{ : } );
else
    [ c, gamma ] = computeEigenfunctions@nlsaKoopmanOperator( ...
        obj, diffOp, parentOpt{ : } );
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tWall0 = tic;

% Compute Dirichlet energies of eigenvectors
eta = computeRegularizingEigenvalues( obj, diffOp );
eta = eta( : );  % ensure eta is a column vector
epsilon = getRegularizationParameter( obj );
E = sum( eta .* abs( c ) .^ 2, 1 );  

% Sort results in order of increasing Dirichlet energy
[ E, idxE ] = sort( E, 'ascend' );
gamma = gamma( idxE );
c = c( :, idxE );
tWall = toc( tWall0 );
fprintf( logId, 'ENGY %2.4f \n', tWall );

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

