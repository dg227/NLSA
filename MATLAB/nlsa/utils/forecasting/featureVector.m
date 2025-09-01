function k = featureVector( phiO, phi, lambda, nPar )
% FEATUREVECTOR Compute feature vector from kernel eigenfunctions.
%
% phi:     Array of size [ nS nL ] containing the values of the eigenfunctions 
%          on the training dataset. nS is the number of training samples and
%          nL the number of eigenfunctions.
%
% phiO:    Array of size [ nSO nL ] containing the out-of-sample values of 
%          the eigenfunctions on the test dataset. nSO is the number of test
%          samples.
%
% lambda:  Row vector of length nL containing the eigenvalues corresponding to 
%          phi.
%
% nPar:    Number of workers for parallel for loop calculation. Calculation 
%          reverts to serial for loop if nPar is set to 0 or is unspecified. 
%
% Output arguments:
%       
% k:       Array of size [ nS nSO ] storing the feature vectors.
%
% Modified 2021/07/05

% Return k as a column vector for a single out-of-sample evaluation point.
if size( phiO, 1 ) == 1
    k = sum( phi .* ( phiO .* lambda ), 2 );
    return
end

% Recursive call for multiple observables
if nargin <= 4 
    nPar = 0;
end

nSO = size( phiO, 1 );
parfor ( iS = 1 : nSO, nPar )
    k( :, iS ) = featureVector( phiO( iS, : ), phi, lambda ); 
end
