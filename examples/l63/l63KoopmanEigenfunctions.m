% SCRIPT TO COMPUTE APPROXIMATE KOOPMAN EIGENFUNCTIONS USING THE NLSA 
% BASIS
%
% Modified 2017/02/23

%% MAIN CALCULATION PARAMETERS AND OPTIONS
experiment = '64k'; 
idxPhi     = 1 : 51; % NLSA eigenfunction indices used for Galerkin
epsilon    = 1E-4;   % diffusion regularization parameter
regMethod  = 'inv';  % diffusion regularization method

%% SCRIPT EXCECUTION OPTIONS
ifRead     = true; % read NLSA eigenfunctions 
ifCalcPsi  = true; % compute Koopman eigenfunctions 

%% BUILD NLSA MODEL, SET UP OUTPUT DIRECTORY
[ model, Pars ] = l63NLSAModel_den( experiment );
outDir = fullfile( getPath( model.diffOp ), ...
                   'koopman', ...
                   sprintf( '%s_eps_%1.3g_idxPhi%i-%i', ...
                            regMethod, epsilon, idxPhi( 1 ), idxPhi( 2 ) ) );

%% READ DATA FROM MODEL
if ifRead
    tic
    disp( 'Eigenfunction data')
    [ phi, mu, eta ] = getDiffusionEigenfunctions( model ); 
    switch regMethod
        case 'log'
            eta = - log( eta );
            outDirPrefix = '';
        case 'neg'
            eta = 1 - eta;
            outDirPrefix = 'neg';
        case 'inv'
            eta = 1 ./ eta - 1;
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
%  lambda: Koopman eigenvalues
%  psi:    Koopman eigenfunctions
%  E:      Dirichlet energies of Koopman eigenfunctions
if ifCalcPsi
    tic
    disp( 'Koopman eigenvalue problem' )
    varPhi = bsxfun( @rdivide, phi, sqrt( etaNorm ) );
    dphi  = 0.5 * ( varPhi( 3 : end, : ) - varPhi( 1 : end - 2, : ) ); 
    phiMu = bsxfun( @times, varPhi, mu );
    v = phiMu( 2 : end - 1, idxPhi )' * dphi( :, idxPhi );
    B = diag( 1 ./ etaNorm( idxPhi ) );
    L = v - epsilon * diag( [ 0 ones( 1, numel( idxPhi ) - 1 ) ] );
    [ c, lambda ] = eig( L, B );
    psiNorm = sum( bsxfun( @rdivide, abs( c ) .^ 2, etaNorm( idxPhi )' ), 1 );
    c = bsxfun( @rdivide, c, sqrt( psiNorm ) );
    lambda = diag( lambda );
    E = min( 1, eta );
    E( 1 ) = 0;
    E = sqrt( sum( bsxfun( @times, abs( c ) .^ 2, E( idxPhi ) ), 1 ) );
    [ E, idxE ] = sort( E, 'ascend' );
    lambda = lambda( idxE );
    omega = imag( lambda );
    c = c( :, idxE );
    psiNorm = psiNorm( idxE );
    psi = varPhi( :, idxPhi  ) * c;
    toc
end
