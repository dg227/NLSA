function [ fPred, cT, fErr2, cT2 ] = analogForecast( f, phi, mu, phiO, nT, ...
                                                     nL, nL2 )
% ANALOGFORECAST Perform analog forecasting of time series data using a basis.
%
% Input arguments:
%
% f:      Training data array of size [ nD, nSF, nR ], where nD is the data 
%         space dimension, nSF the number of response samples in each 
%         realization (ensemble member for training), and nR the number of 
%         training realizations (ensemble size). 
%
% phi:    Basis function array of size [ nS * nR, nLMax ], where nLMax is the 
%         number of available basis functions, and nS the number of covariate
%         values in each realization. nS should be less than or equal to nSF. 
%
% mu:     Inner product weight array of size [ nS * nR, 1 ]. The phi's 
%         are assumed to be orthonormal with respect to the mu inner product. 
%
% phiO:   Array of size [ nSO, nLMax ] containing the out-of-sample values of 
%         the basis functions at forecast initialization. nSO is the number of 
%         test data. 
% 
% nT:     Number of prediction timesteps, including the forecast initialization
%         time. If nS + nT - 1 > nSF, the training data are padded by zeros. 
%
% nL:     Number of basis functions used for forecasting. Must be <= nLMax. 
%
% nL2:    Number of basis functions used for conditional variance estimation.
%         Must be <= nLMax.
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
% fErr2:  Array of size [ nD, nT, nSO ] containing the estimated forecast 
%         square errors.
%
% cT2:    Array of size [ nD, nT, nL2 ] containing the expansion coefficients
%         of the square error estimator in the phi basis. 
%
% Modified 2020/08/13

% Get array dimensions
[ nD, nSF, nR ] = size( f );
[ nSR, nLMax ] = size( phi ); 
nS = nSR / nR; 
nSO = size( phiO, 1 );

% If needed, set default values for basis function numbers
if nargin < 6
   nL = nLMax; 
end
if nargin < 7
    nL2 = nL;
end

% If needed, pad the training data with zeros
nSPad = nS + nT - 1 - nSF;
if nSPad > 0
    f = cat( 2, f, zeros( nD, nSPad, nR ) );
    nSF = nSF + nSPad;
end

% Put training data in appropriate form for time shift (temporal index is
% last) 
f = permute( f, [ 1 3 2 ] ); % size [ nD nR nSF ]
f = reshape( f, [ nD * nR, nSF ] );

% Create time-shifted copies of the input signal
fT = lembed( f, [ nT, nT + nS - 1 ], 1 : nT );

% Put in appropriate form to take inner product with basis functions
fT = reshape( fT, [ nD nR nT nS ] );
fT = permute( fT, [ 1 3 4 2 ] ); % size [ nD nT nS nR ]
fT = reshape( fT, [ nD * nT, nS * nR ] );

% Compute expansion coefficients of the time-shifted signal with respect to the
% basis
cT = fT * ( phi( :, 1 : nL ) .* mu );

% Evaluate forecast using out-of-sample values of the basis functions
fPred = cT * phiO( :, 1 : nL ).';
fPred = reshape( fPred, [ nD nT nSO ] );

% Reshape expansion coefficient array
cT = reshape( cT, [ nD nT nL ] );

% Quick return if no error etimation is requested
if nargout < 3
    return
end

% Compute square forecast error in the training data (in-sample error)
cT = reshape( cT, [ nD * nT, nL ] );
fT = ( fT - cT * phi( :, 1 : nL ).' ) .^ 2; 

% Compute expansion coefficients of the square forecast error with respect to
% the basis
cT2 = fT * ( phi( :, 1 : nL2 ) .* mu );   

% Evaluate square forecast error estimate using out-of-sample values of the
% basis functions
fErr2 = cT2 * phiO( :, 1 : nL2 ).'; 
fErr2 = reshape( fErr2, [ nD nT nSO ] );

% Reshape expansion coefficient array
cT2 = reshape( cT2, [ nD nT nL2 ] );




