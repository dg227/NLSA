function c = structureConstants( phi, mu, nL, nL2, nPar )
% STRUCTURECONSTANTS Compute the structure constnants associated with 
% multiplication of basis elements.
%
% Input arguments:
%
% phi:     Array of size [ nS, nLMax ] where nS is the number of samples and 
%          nLMax the total number of available basis functions.
%
% mu:      Inner product weight array of size [ nS, 1 ]. The phi's 
%          are assumed to be orthonormal with respect to the mu inner product. 
%
% nL, nL2: Positive integers <= nLMax determining the size of the structure
%          constant array. If nL is unspecified or empty, it is set to nLMax. 
%          If nL2 is unspecified or empty, it is set to nL.
%
% nPar:    Number of workers for parallel for loop calculation. Calculation 
%          reverts to serial for loop if nPar is set to 0 or is unspecified. 
%
% Output arguments:
%
% c:       Array of size [ nL, nL, nL2 ] containing the structure constants.
%          Specifically, c( i, j, k ) = < phi( :, i ), phi( :, j ) phi( :, k ) >
%          where < , > denotes the mu-weighted inner product.
%
% Modified 2021/03/06.

if nargin == 2 || isempty( nL )
    nL = size( phi, 2 );
end

if nargin <+ 3 || isempty( nL2 )
    nL2 = nL:
end

if nargin <= 4 
    nPar = 0;
end

% Compute mu-weighted basis functions
phiMu = ( phi( :, 1 : nL ) .* mu )';

% Compute structure constants
c = zeros( nL, nL, nL2 );
parfor( i = 1 : nL2, nPar )
    c( :, :, i ) = phiMu * ( phi( :, 1 : nL ) .* phi( :, i ) );
end
