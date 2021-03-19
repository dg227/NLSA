function [ p, rhoOut ] = quantumForecast( V, U, rho, nTF, nTOut, nPar )
% QUANTUMFORECAST Perform quantum mechanical probabilistic forecasting.
%
% Input arguments:
%
% V:       Array of size [ nL, nL ] containing, in its columns, eigenvectors of
%          the multiplication operator representing the forecast observable. 
%
% U:       An [ nL, nL ] matrix or a cell vector of [ nL, nL ] matrices 
%          representing the Koopman operator for the desired forecast time
%          steps. 
%
% rho:     An [ nL, nL ] density matrix representing the initial condition. 
%
% nTF:     Number of forecast timesteps.
%
% nTOut:   Number of timesteps to return the time-evolved density matrix. 
%
% nPar:    Number of workers for parallel for loop calculation. Calculation 
%          reverts to serial for loop if nPar is set to 0 or is unspecified. 
%
% Output arguments:
%
% p:       Array of size [ nL, nTF ] containing the forecast probabilities. 
%          Specifically, p( i, j ) is the probability to obtain an observation
%          equal to the eigenvalue corresponding to eigenvector V( :, i ) at the
%          j-th timestep.
%
% rhoOut:  Density matrix at nTOut steps. This output is intended to be used
%          in conjunction with observational state update to perform data 
%          assimilation.
%
% Modified 20201/03/02

% Set default arguments
if nargin < 6 
    nPar = 0;
end

if nargin < 5
    nTOut = [];
end

if ~iscell( U )
    U = { U };
end

nL = size( V, 1 );

p = zeros( nL, nTF );

% Loop over forecast steps
parfor( iT = 1 : nTF, nPar )
   rhoT = U{ iT }' * rho * U{ iT };
   rhoT = rhoT / trace( rhoT );
   if iDT == nTOut
       rhoOut = rhoT;
   end
   for iL = 1 : nL
       p( iL, iT ) = V( :, iL )' * rhoT * V( :, iL );
   end
end

if isempty( nTOut )
    rhoOut = [];
end

