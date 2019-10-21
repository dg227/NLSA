% SCRIPT TO COMPUTE APPROXIMATE KOOPMAN EIGENFUNCTIONS USING THE NLSA 
% BASIS
%
% Modified 2017/02/23

%% MAIN CALCULATION PARAMETERS AND OPTIONS
experiment = 'ip_sst_control'; 
idxPhi     = 1 : 1001; % NLSA eigenfunction indices used for Galerkin
epsilon    = 1E-3;   % diffusion regularization parameter
regMethod  = 'inv';  % diffusion regularization method

%% SCRIPT EXCECUTION OPTIONS
ifRead     = true; % read NLSA eigenfunctions 
ifCalcPsi  = true; % compute Koopman eigenfunctions 

%% BUILD NLSA MODEL, SET UP OUTPUT DIRECTORY
[ model, Pars ] = climateNLSAModel( experiment );
%[ model, Pars ] = climateNLSAModel_ose( experiment );
%outDir = fullfile( getPath( model.diffOp ), ...
%                   'koopman', ...
%                   sprintf( '%s_eps_%1.3g_idxPhi%i-%i', ...
%                            regMethod, epsilon, idxPhi( 1 ), idxPhi( 2 ) ) );

%% READ DATA FROM MODEL
if ifRead
    tic
    disp( 'Eigenfunction data')
    [ phi, mu, lambda ] = getDiffusionEigenfunctions( model ); 
    switch regMethod
        case 'log'
            eta = - log( lambda );
            outDirPrefix = '';
        case 'neg'
            eta = 1 - lambda;
            outDirPrefix = 'neg';
        case 'inv'
            eta = 1 ./ lambda - 1;
            outDirPrefix = 'inv';
    end
    eta = eta / eta( 2 );
    etaNorm = eta';
    etaNorm( 1 ) = 1;
    nS = numel( mu );
    toc
end

%% EIGENVALUE PROBLEM FOR KOOPMAN GENERATOR 
%  Output arrays are as follows:
%  gamma: Koopman eigenvalues
%  omega: Koopman eigenfrequencies (in units of 1/year)
%  psi:   Koopman eigenfunctions
%  E:     Dirichlet energies of Koopman eigenfunctions
if ifCalcPsi
    tic
    disp( 'Koopman eigenvalue problem' )
    varPhi = bsxfun( @rdivide, phi, sqrt( etaNorm ) );
    dphi  = 0.5 * ( varPhi( 3 : end, : ) - varPhi( 1 : end - 2, : ) ) * 12; 
    phiMu = bsxfun( @times, varPhi, mu );
    v = phiMu( 2 : end - 1, idxPhi )' * dphi( :, idxPhi );
    B = diag( 1 ./ etaNorm( idxPhi ) );
    L = v - epsilon * diag( [ 0 ones( 1, numel( idxPhi ) - 1 ) ] );
    [ c, gamma ] = eig( L, B );
    psiNorm = sum( bsxfun( @rdivide, abs( c ) .^ 2, etaNorm( idxPhi )' ), 1 );
    c = bsxfun( @rdivide, c, sqrt( psiNorm ) );
    gamma = diag( gamma );
    E = sqrt( sum( abs( c( 2 : end, : ) ) .^ 2, 1 ) );
    [ E, idxE ] = sort( E, 'ascend' );
    gamma = gamma( idxE );
    omega = imag( gamma );
    per = 2 * pi ./ omega;
    c = c( :, idxE );
    psiNorm = psiNorm( idxE );
    psi = varPhi( :, idxPhi  ) * c;
    toc
end
