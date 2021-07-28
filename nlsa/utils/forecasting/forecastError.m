function [ fErr, fRMSE, fPC ] = forecastError( fTrue, fPred, nTO )
% FORECASTERROR Compute forecast errors and root mean square error (RMSE), 
% pattern correlation (PC) forecast skill scores. 
%
% Input arguments:
% 
% fTrue: True signal array of size [ nD, nST ], where nD is the data space 
%        dimension and nST the number of test samples.
%
% fPred: Predicted signal array of size [ nD, nTF, nSO ], where nTF is the 
%        number of forecast timsteps (including zero) and nSO the number of 
%        initial conditions.
%
% nTO:   Number of timesteps between forecast initializations. It is set equal
%        to 1 if unspecified by the caller.  
%
% Output arguments:
%
% fRMSE, fPC: RMSE and PC arrays of size [ nD, nTF ].
%
% fErr:  Error array of size [ nD, nTF, nSO ] 
%
% Modified 2021/02/13

if nargin < 3
    nTO = 1;
end

% Get array sizes
[ nD, nST ]     = size( fTrue );
[ ~, nTF, nSO ] = size( fPred );

% Initialize error array, fill in available samples from true signal.
% The error array at the end of this calculation should have size [ nD nTF nSO ]
nSO2    = nSO + ( nSO - 1 ) * ( nTO - 1 ); 
nSTLim  = min( nSO2 + nTF - 1, nST );
nSCount = nSTLim - nTF + 1; 

fErr                   = nan( nD * nTF, nSO2 );
fErr( :, 1 : nSCount ) = lembed( fTrue, [ nTF, nSTLim  ], 1 : nTF );
fErr                   = fErr( :, 1 : nTO : end ); 

% Reshape in appropriate form for comparison with fPred
fErr = reshape( fErr, [ nD nTF nSO ] ); 

% Compute forecast errors
fErr = fErr - fPred;  

if nargout == 1
    return
end

% Compute RMSE 
fRMSE = sqrt( mean( fErr( :, :, 1 : nSCount ) .^ 2, 3 ) );

if nargout == 2
    return
end

% Compute PC score
fPC = nan( nD, nTF );
if nSCount > 0 
    for iT = 1 : nTF
        x = fTrue( :, iT : nSCount + iT - 1 );
        y = squeeze( fPred( :, iT, 1 : nSCount ) );
        if nD == 1
            y = y.';
        end
        fPC( :, iT ) = pc( x, y ); 
    end
end

