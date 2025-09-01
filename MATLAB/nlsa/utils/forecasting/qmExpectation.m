function aExp = qmExpectation( A, rho, nPar )
% QMEXPECTATION Compute expectation of quantum mechanical observables
%
% Input arguments
%
% A:       Matrix of size [ nL nL ] or array of size [ nL nL nA ]. nL is the 
%          dimension of the Hilbert space (number of basis functions), and nA
%          is the number of observables.
%
% rho      Matrix of size [ nL nL ] or row vector of length nL representing 
%          the quantum state. If rho is a matrix, then the state is assumed to 
%          be mixed and rho represents the density matrix. If rho is a vector, 
%          then the state is assumed to be pure and rho represents the 
%          wavefunction. Note that our convention is to represent rho with a 
%          row vector as it is "dual" to classical observables (functions) 
%          which are represented by column vectors.
%
% nPar:    Number of workers for parallel for loop calculation. Calculation 
%          reverts to serial for loop if nPar is set to 0 or is unspecified. 
%
% Output arguments:
% 
% aExp:    A scalar or a vector of length nA containing the expectation values
%          of the observable(s) A on the state rho. 
%
% Modified 2021/07/18

if ndims( A ) == 2
    if isrow( rho )
        aExp = rho * A * rho';
    else
        aExp = tr( rho * A );
    end
    return
end

% Recursive call for multiple observables
if nargin <= 3 
    nPar = 0;
end

nA = size( A, 3 );
aExp = zeros( nA, 1 );

parfor( iA = 1 : nA, nPar )
    aExp( iA ) = qmExpectation( A( :, :, iA ), rho );
end
