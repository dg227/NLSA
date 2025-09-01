function [epsilonOpt, Info] = autotune(d2, h, Opts)

    arguments
        d2 (:, 1) {mustBeNumeric} 
        h  (1, 1) function_handle

        Opts.base     (1, 1) {mustBeNumeric} = 2
        Opts.exponent (1, 2) {mustBeNumeric} = [-10 10]
        Opts.n        (1, 1) {mustBeInteger, mustBePositive} = 200
        Opts.nBatch   (1, 1) {mustBeInteger, mustBePositive} = 1
        Opts.nPar     (1, 1) {mustBeInteger, mustBeNonnegative} = 0
    end

    exponents = linspace(Opts.exponent(1), Opts.exponent(2), Opts.n); 
    epsilons = Opts.base .^ exponents; 
    dexp = exponents(2) - exponents(1);

    if isscalar(epsilons)
        epsilonOpt = epsilon;
        Info.logE  = log(logE); 
        Info.iOpt  = [];
        Info.dEst  = [];
        Info.dLogS = [];
    return
end
    
    kSums = zeros(1, Opts.n);

    m = numel(d2);
    idx = decomp1d(m, Opts.nBatch);

    % for iBatch = 1 : Opts.nBatch
    parfor(iBatch = 1 : Opts.nBatch, Opts.nPar)
        i1 = idx(iBatch);
        i2 = idx(iBatch + 1);
        gd = gpuDevice;
        % [iBatch gd.Index]
        kSums = kSums + sum(h(d2(i1 : i2) ./ epsilons .^ 2), 1);
    end

    logS = log(kSums / m);
    logE = log(epsilons);

    % dLogS = (kSums(3 : end) - kSums(1 : end - 2)) ./ kSums(2 : end - 1) ...
    %       ./ (2 * dexp * log(2));
    dLogS =  (logS(3 : end) - logS(1 : end - 2 )) ...
          ./ (logE(3 : end) - logE(1 : end - 2));
    [dEst, iOpt] = max(dLogS);
    epsilonOpt = exp(logE(iOpt + 1));

    Info = struct('iOpt',  iOpt, ...
                  'dEst',  dEst, ...
                  'logE',  logE, ...
                  'logS',  logS, ...
                  'dLogS', dLogS);
end

