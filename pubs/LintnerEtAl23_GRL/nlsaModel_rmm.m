function [model, In, Out] = nlsaModel_rmm(P)
% Create NLSA model for Koopman analysis of RMM data
%
% P: Input parameter structure. 
%
% Modified 2023/06/08

    names = {
        'pth' 
        'dataset'
        'file'
        'tFormat'
        'tStart'
        'tLim'
        'xLim'
        'yLim'
        'var'
        'embWindow'
        'kernel'
        'ifDen'
        };

    args = namedargs2cell(filterField(P, names));
    [model, In, Out] = nlsaModel(args{:}); 
end


function [model, In, Out] = nlsaModel(P)
    arguments
        P.pth       (1, :) {mustBeText} = './data/raw'
        P.dataset   (1, :) {mustBeText} = 'LGPS23'
        P.file      (1, :) {mustBeText} = 'RMM_data.mat'
        P.tFormat   (1, :) {mustBeTextScalar} = 'yyyymmdd'
        P.tStart    (1, :) {mustBeTextScalar} = '19980101'
        P.tLim      (1, 2) {mustBeText} = {'19980101' '20191230'}
        P.xLim      (1, 2) {mustBeNumeric} = [40 290]
        P.yLim      (1, 2) {mustBeNumeric} = [-15 15]
        P.var       (1, :) {mustBeText} = 'RMM_data'
        P.embWindow (1, 1) {mustBePositive, mustBeInteger} = 1
        P.kernel    (1, :) {mustBeTextScalar} = 'l2'
        P.ifDen     (1, 1) {mustBeNumericOrLogical} = false
    end

    experiment = experimentStr_rmm(P);

    switch experiment
        case 'LGPS23_RMM_19980101-20191230_emb64_cone'
            % Delay-embedding/finite-difference parameters; in-sample data
            In.Src(1).nXB       = 2;          % samples before main interval
            In.Src(1).nXA       = 2;          % samples after main interval
            In.Src(1).fdOrder   = 4;          % finite-difference order 
            In.Src(1).fdType    = 'central';  % finite-difference type
            In.Src(1).embFormat = 'overlap';  % storage format 

            % Batches to partition the in-sample data
            In.Res(1).nB    = 2; % partition batches
            In.Res(1).nBRec = 2; % batches for reconstructed data

            % NLSA parameters; in-sample data 
            In.nN         = 1500;       % nearest neighbors; defaults to max. value if 0
            In.tol        = 0;          % 0 distance threshold (for cone kernel)
            In.zeta       = 0.995;      % cone kernel parameter 
            In.coneAlpha  = 0;          % velocity exponent in cone kernel
            In.nNS        = In.nN;      % nearest neighbors for symmetric distance
            In.diffOpType = 'gl_mb_bs'; % diffusion operator type
            In.epsilon    = 2;          % kernel bandwidth parameter 
            In.epsilonB   = 2;          % kernel bandwidth base
            In.epsilonE   = [-40 40]; % kernel bandwidth exponents 
            In.nEpsilon   = 200;        % number of exponents for bandwidth tuning
            In.alpha      = 0.5;        % diffusion maps normalization 
            In.nPhi       = 151;        % diffusion eigenfunctions to compute
            In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
            In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
            In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
            In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

            % Koopman generator parameters; in-sample data
            In.koopmanOpType = 'diff';     % Koopman generator type
            In.koopmanFDType  = 'central'; % finite-difference type
            In.koopmanFDOrder = 4;         % finite-difference order
            In.koopmanDt      = 1;         % sampling interval (in months)
            In.koopmanAntisym = true;      % enforce antisymmetrization
            In.koopmanEpsilon = 1.0E-3;      % regularization parameter
            In.koopmanRegType = 'inv';     % regularization type
            In.idxPhiKoopman  = 1 : 101;   % diffusion eigenfunctions used as basis
            In.nPhiKoopman    = numel(In.idxPhiKoopman);        % Koopman eigenfunctions to compute
            In.nKoopmanPrj    = In.nPhiKoopman; % Koopman eigenfunctions for projection
            In.idxKoopmanRec = {[2 3]};

        otherwise
           error ('Invalid experiment') 
    end

    % Prepare source and target components
    In.Res(1).experiment = P.dataset;
    In.tFormat           = P.tFormat; 
    In.Res(1).tLim       = P.tLim; 
    In.Src(1).field      = P.var;
    In.Src(1).xLim       = P.xLim; 
    In.Src(1).yLim       = P.yLim; 
    In.Src(1).idxE       = 1 : P.embWindow; % delay-embedding indices 
    In.lDist             = P.kernel; % local distance function

    % Check if we are doing out-of-sample extension
    ifOse = exist('Out', 'var');

    % Serial date numbers for in-sample data
    for iR = 1 : numel(In.Res)
        limNum = datenum(In.Res(iR).tLim, In.tFormat);
        In.Res(iR).tNum = limNum(1) : limNum(2);
        nS = numel(In.Res(iR).tNum);
    end

    % Serial date numbers for out-of-sample data
    if ifOse
        for iR = 1 : numel(Out.Res)
            limNum = datenum(Out.Res(iR).tLim, Out.tFormat);
            Out.Res(iR).tNum = limNum(1) : limNum(2);
            nS = numel(Out.Res(iR).tNum);
        end
    end

    % Construct NLSA model
    if ifOse
        args = {In Out};
    else
        args = {In};
    end
    [model, In, Out] = climateNLSAModel(args{:});
end
