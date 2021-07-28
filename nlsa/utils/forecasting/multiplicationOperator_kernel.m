function M = multiplicationOperator_kernel( phiO, lambda, c, nPar )
% MULTIPLICATIONOPERATOR_KERNEL Compute multiplication operator associated with
% kernel feature vector. 
%
% Input arguments:
%
% phiO:    Vector of length nL2 containing the (out-of-sample) values of kernel
%          features, used as expansion coefficients in a Mercer sum.   
%
% lambda:  Vector of lambda nL2 containing the coefficients (weights) in the 
%          Mercer expansion.
%
% c:       Array of size [ nL, nL, nL2 ] containing the multiplicative 
%          structure constants associated with the kernel features (see 
%          function structureConstants).
%
% nPar:    Number of workers for parallel for loop calculation. Calculation 
%          reverts to serial for loop if nPar is set to 0 or is unspecified. 
% 
% Output arguments:
%
% M:       Array of size [ nL, nL ] containing the matrix elements of the 
%          multiplication operator associated with the kernel features.
%
% Modified 2021/03/07

if nargin <= 4 
    nPar = 0;
end

nL  = size( c, 1 );
nL2 = size( c, 3 ); 

M = zeros( nL );

parfor ( i = 1 : nL2, nPar )
    M = M + phiOL( iL ) * lambda( iL ) * c( :, :, iL );  
end
