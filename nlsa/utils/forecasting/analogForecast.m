function [ fPred, cT ] = analogForecast( f, phi, mu, phiO, nT )
% ANALOGFORECAST Perform analog forecasting of time series data using a basis
%
% Input arguments:
%
% f:      Training data array of size [ nD, nSF, nR ], where nD is the data 
%         space dimension, nSF the number of samples in each realization
%         (ensemble member for training), and nR the number of training 
%         realizations (ensemble size). 
%
% phi:    Basis function array of size [ nS * nR, nL ], where nL is the number
%         of eigenfunctions, and nS the number of eigenfunction values 
%         in each realization. nS should be less than or equal to nSF. 
%
% mu:     Inner product weight array of size [ nS * nR, 1 ]. The phi's 
%         are assumed to be orthonormal with respect to the mu inner product. 
%
% phiO:   Array of size [ nSO, nL ] containing the out-of-sample values of the
%         basis functions at forecast initialization. nSO is the number of 
%         test data. 
% 
% nT:     Number of prediction timesteps, including the forecast initialization
%         time. If nS + nT - 1 > nSF, the training data are padded by zeros. 
%
% Output arguments:
% 
% fPred:  Forecast data array of size [ nD, nT, nSO ]. f( :, iT, iSO ) is the 
%         nD-dimensional forecast vector for lead time iT and the iSO-th 
%         initial condition.
%
% cT:     Array of size [ nD, nT, nL ] containing the expansion coeffcients of
%         the target (prediction) function in the phi basis. 
%
% Modified 2020/08/06

% Get array dimensions
[ nD, nSF, nR ] = size( f );
[ nSR, nL ] = size( phi ); 
nS = nSR / nR; 
nSO = size( phiO, 1 );

% If needed, pad the training data with zeros
nSPad = nS + nT - 1 - nSF;
if nSPad > 0
    f = cat( 2, f, zeros( nD, nSPad, nR ) );
    nSF = nSF + nSPad;
end

% Put training data in appropriate form for time shift (temporal index is
% last) 
f = permute( f, [ 1 3 2 ] ); % size [ nD nR nSF ]
f = reshape( fTtrain, [ nD * nR, nSF ] );

% Create time-shifted copies of the input signal
fT = lembed( f, [ nT, nT + nS - 1 ], 1 : nT );

% Put in appropriate form to take inner product with basis functions
fT = reshape( fT, [ nD nR nT nS ] );
fT = permute( fT, [ 1 3 4 2 ] ); % size [ nD nT nS nR ]
fT = reshape( fT, [ nD * nT, nS * nR ] );

% Compute expansion coefficients of the time-shifted signal with respec to the
% basis
cT = fT * phi .* mu;

% Evaluate forecast using out-of-sample values of the eigenfunctions
fPred = cT * phiO;

% Reshape coefficient array
if nargout > 1
    cT = reshape( cT, [ nD nT nL ] );
end
