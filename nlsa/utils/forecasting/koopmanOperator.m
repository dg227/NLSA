function U = koopmanOperator(q, phi, mu, Opts)
    % KOOPMANOPERATOR Compute an approximation of the Koopman operator in a basis 
    % of observables.
    %
    % Input arguments:
    %
    % q:    Scalar or vector of non-negative integers containing the shifts to 
    %       apply.
    %
    % phi:  Array of size [ nS nL ] where nS is the number of samples and nL the 
    %       number of basis functions.
    %
    % mu:   Inner product weight array of size [ nS, 1 ]. The phi's 
    %       are assumed to be orthonormal with respect to the mu inner product. 
    %
    % nPar: Number of workers for parallel for loop calculation. Calculation 
    %       reverts to serial for loop if nPar is set to 0 or is unspecified. 
    %
    % Output arguments:
    %
    % U:    If q is scalar, U is an [ nL, nL ] sized matrix representing the 
    %       Koopman operator in the phi basis. If q is a vector, U is an array of 
    %       size [ nL nL nQ ], where nQ is the length of q, such that U(:, :, i) 
    %       contains the Koopman operator matrix for shift q(i).
    %
    % Modified 2022/11/11.

    arguments
        q (:, 1) {mustBeInteger, mustBeNonnegative}
        phi (:, :) {mustBeNumeric}
        mu (:, 1) {mustBeNumeric}
     
        Opts.nPar (1, 1) {mustBeInteger, mustBeNonnegative} = 0
        Opts.polar (1, 1) {mustBeNumericOrLogical} = 0
    end

    % Return U as a matrix for a single shift q
    if isscalar(q)
        U = phi(1 : end - q, :)' * (phi(q + 1 : end, :) .* mu(1 : end - q));
        if Opts.polar
            disp('Doing SVD')
            [u, ~, v] = svd(U);
            U = u * v';
        end
    else
        % Recursive call for a vector of shifts
        nL = size(phi, 2);
        nQ = numel(q);
        U  = zeros(nL, nL, nQ);

        parfor(iQ = 1 : nQ, Opts.nPar)
            U(:, :, iQ) = koopmanOperator(q(iQ), phi, mu, polar=Opts.polar);
        end
    end
end
