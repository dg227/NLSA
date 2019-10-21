% SCRIPT TO COMPUTE APPROXIMATE KOOPMAN EIGENFUNCTIONS USING THE NLSA 
% BASIS

%% MAIN CALCULATION PARAMETERS AND OPTIONS
experiment = 'lo_res'; 
idxPhi     = 1 : 200; % NLSA eigenfunction indices used for Galerkin
epsilon    = 7E-4;   % diffusion regularization parameter
regMethod  = 'inv';  % diffusion regularization method

%% SCRIPT EXCECUTION OPTIONS
ifRead     = false; % read NLSA eigenfunctions 
ifCalcPsi  = true; % compute Koopman eigenfunctions 

%% BUILD NLSA MODEL, SET UP OUTPUT DIRECTORY
[ model, Pars ] = clausNLSAModel_den( experiment );

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
    v = phiMu( 2 : end - 1, idxPhi )' * dphi( :, idxPhi ) / ( 1 / 8 * Pars.nDT );;
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
    psi = varPhi( :, idxPhi  ) *  c;
    toc
end
