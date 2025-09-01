function M = multiplicationOperator( f, phi, mu, nPar )
% MULTIPLICATIONOPERATOR Compute multiplication operators associated with 
% observables in a basis. 
%
% Input arguments:
%
% f:       Array of size [ nS nF ], where nS is the number of samples and 
%          nF the number of observables, containing the observable values.
%
% phi:     Array of size [ nS nL ], where nL is the number of basis functions,
%          containing the values of the basis functions.
%
% mu:      Inner product weight array of size [ nS 1 ]. The phi's 
%          are assumed to be orthonormal with respect to the mu inner product. 
%
% nPar:    Number of workers for parallel for loop calculation. Calculation 
%          reverts to serial for loop if nPar is set to 0 or is unspecified. 
%
% Output arguments:
% 
% M:       If nF is equal to 1, M is an array of size [ nL nL ] containting 
%          the matrix elements of the multiplication operator. Specifically, 
%          M( i, j ) = < phi( :, i ), f( : ) phi( :, j ) >. where < , > 
%          denotes mu-weighted inner product. If nF is greater than 1, M is a 
%          an array of size [ nL nL nF ]  such that M( :, :, i ) contains the 
%          multiplication operator  matrix for the observable in the i-th 
%          column of f.
%
% Modified 2021/07/05


% Return M as a matrix for a single (scalar) observable f
if iscolumn( f )
    M = phi' * ( phi .* ( f .* mu ) );
    return
end

% Recursive call for multiple observables
if nargin <= 3 
    nPar = 0;
end

nL = size( phi, 2 );
nF = size( f, 2 );
M = zeros( nL, nL, nF ); 

parfor( iF = 1 : nF, nPar )
    M( :, :, iF ) = multiplicationOperator( f( :, iF ), phi, mu, nPar );
end
