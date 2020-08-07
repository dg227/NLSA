function [ fErr, fRMSE, fPC ] = forecastError( fTrue, fPred )
% FORECASTERROR Compute forecast errors and root mean square error (RMSE), 
% pattern correlation (PC) forecast skill scores. 
%
% Input arguments:
% 
% fTrue: True signal array of size [ nD, nST ], where nD is the data space 
%        dimension and nST the number of test samples.
%
% fPred: Predicted signal array of size [ nD, nT, nSO ], where nT is the number
%        of forecast timsteps (including zero) and nSO the number of initial
%        conditions.
%
% Output arguments:
%
% fRMSE, fPC: RMSE and PC arrays of size [ nD, nT ].
%
% fErr:       Array of size [ nD, nT, nSO ] 
%
% Modified 2020/08/06

% Get array sizes
[ nD, nST ] = size( fTrue );
[ ~, nT, nSO ] = size( fPred );

% Initialize error array, fill in available samples from true signal  
fErr = nan( nD * nT, nSO );
nSTLim = min( nSO + nT - 1, nST );
nSCount = nSTLim - nT + 1; 
fErr( :, 1 : nSCount ) = lembed( fTrue, [ nT, nSTLim  ], 1 : nT );

% Reshape in appropriate form for comparison with fPred
fErr = reshape( fErr, [ nD nT nSO ] ); 

% Compute forecast errors
fErr = fT - fPred;  

if nargout == 1
    return
end

% Compute RMSE 
fRMSE = sqrt( mean( fErr( :, :, 1 : nSCount .^ 2, 3 ) );

if nargout == 2
    return
end

% Compute PC score
fPC = zeros( nD, nT );
for iT = 1 : nT
    fPC( :, iT ) = pc( fTrue( :, iT : nSCount + iT - 1 ), ...
                       squeeze( fPred( :, iT, 1 : nSCount ) ) ); 
end

