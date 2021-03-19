function U = koopmanOperator( phi, mu, q, nL, nPar )
% KOOPMANOPERATOR Compute an approximation of the Koopman operator in a basis 
% of observables.
%
% Input arguments:
%
% phi:  Array of size [ nS, nLMax ] where nS is the number of samples and 
%       nLMax the total number of available basis functions.
%
% mu:   Inner product weight array of size [ nS, 1 ]. The phi's 
%       are assumed to be orthonormal with respect to the mu inner product. 
%
% q:    Vector of non-negative integers containing the shifts to apply.
%
% nL:   Number of basis functions to employ for operator approximation. This
%       argument is set to nLMax if not speciried or empty. 
%
% nPar: Number of workers for parallel for loop calculation. Calculation 
%       reverts to serial for loop if nPar is set to 0 or is unspecified. 
%
% Output arguments:
%
% U:    If q is scalar, U is an [ nL, nL ] sized matrix representing the 
%       Koopman operator in the phi basis. If q is a vector, U is a cell 
%       vector such that U{ i } containts the Koopman operator matrix for 
%       shift q( i ).
%
% Modified 2021/03/01.

if nargin == 3 || isempty( nL )
    nL = size( phi, 2 );
end

% Return U as a matrix for a single shift q
if isscalar( q )
    U = phi( 1 : end - q, 1 : nL )' ...
      * ( phi( q + 1 : end, 1 : nL ) .* mu( 1 : end - q) );
    return
end

% Recursive call for a vector of shifts
if nargin <= 4 
    nPar = 0;
end

nQ = numel( q );
U = cell( 1, nQ );

parfor( iQ = 1 : nQ, nPar )
    U{ iQ } = koopmanOperator( phi, mu, q( iQ ), nPar );
end
