function M = multiplicationOperator( f, phi, mu, nL )
% MULTIPLICATIONOPERATOR Compute multiplication operator associated with 
% an observable in a basis. 
%
% Input arguments:
%
% f:       Column vector of length nS, where nS is the number of samples, 
%          containing the values of observable f.
% phi:     Array of size [ nS, nLMax ], where nLMax is the total number of 
%          available basis functions.
%
% mu:      Inner product weight array of size [ nS, 1 ]. The phi's 
%          are assumed to be orthonormal with respect to the mu inner product. 
%
% nL:      Positive integer <= nLMax determining the number of basis functions
%          employed in the construction of the multiplication operator. If nL
%          is unspecified, it is set to nLMax.
%
% Output arguments:
% 
% M:       Array of size [ nL, nL ] containting the matrix elements of the 
%          multiplication operator. Specifically, 
%          M( i, j ) = < phi( :, i ), f( : ) phi( :, j ) >. where < , > 
%          denotes mu-weighted inner product. 
%
% Modified 2021/03/01


if nargin == 3
    nL = size( phi, 2 );
end

M = phi( :, 1 : nL )' * ( phi( :, 1 : nL ) .* ( f .* mu ) );


