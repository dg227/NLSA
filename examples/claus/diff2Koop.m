% Create NLSA model with Koopman eigenfunctions instead of diffusion eigenfunctions
%
% This script assumes that a set of Koopman eigenfunctions psi has been 
% computed using the script clausKoopmanEigenfunctions 

% Koopman eigenfunctions to store in new NLSA model 
% The first eigenfunction should not be 1 since the constant eigenfunction is 
% stored by default. 
idxPsi = [ 2 : 12 ]; 

% Koopman eigenfunctions to reconstruct by new NLSA model
% The convention here is that odd values of idxPsiRec reconstruct the 
% real part of Koopman eigenfunction ( idxPsiRec + 1 ) / 2, and even 
% values of idxPsiRec reconstruct the imaginary part of Koopman eigenfunction
% idxPsiRec / 2. For instance, to reconstruct Koopman eigenfunctions 1 and 2 
% set idxPsiRec = [ 1 2 3 4 ]. 
idxPsiRec = [ 1 2 3 4 ];   

% The variables regMethod, epsilon, and idxPhi should be set by the script
% clausKoopmanEigenfunctions
phiK = zeros( size( psi, 1 ), 2 * numel( idxPsi ) + 1 );
phiK( :, 1 ) = 1;
phiK( :, 2 : 2 : end - 1 ) = real( psi( :, idxPsi ) );
phiK( :, 3 : 2 : end ) = imag( psi( :, idxPsi ) );
phiK = bsxfun( @times, phiK, sqrt( mu ) );

[ modelK, InK ] = clausNLSAModel_koopman( 'lo_res', regMethod, epsilon, ...
                                          idxPhi, idxPsi, idxPsiRec );
setEigenfunctions( modelK.diffOp, phiK, mu );

