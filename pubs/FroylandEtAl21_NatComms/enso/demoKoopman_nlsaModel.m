function [model, In] = demoKoopman_nlsaModel(NLSA)
% ENSOLIFECYCLE_NLSAMODEL Construct NLSA model for analysis of ENSO lifecycle
% 
% Input arguments:
%
% NLSA: Structure with the main parameters of the data analysis experiment. 
%       NLSA is converted to a string identifier using the 
%       demoKoopman_experiment function. 
%
% Output arguments:
%
% model: Constructed model, in the form of an nlsaModel object.  
% In:    Structure with in-sample model parameters. 
% Out:   Structure with out-of-sample model parameters (optional). 
%
% This function creates the parameter structures In and Out, which are then 
% passed to function climateNLSAModel to build the model.
%
% The constructed NLSA models have the following target components (used for
% composite lifecycle analysis):
%
% Component 1:  Nino 3.4 index
% 
% Modified 2029/09/14

experiment = demoKoopman_experiment(NLSA); 


switch experiment

% ERSSTV4 data, last 50 years, Indo-Pacific SST input, 4-year delay embeding 
% window, cone kernel  
case 'ersstV4_197001-202002_IPSST_emb48_cone'
    
    % Dataset specification 
    In.Res(1).experiment = NLSA.dataset;
    
    % Time specification 
    In.tFormat        = 'yyyymm';               % time format
    In.Res(1).tLim  = NLSA.period;            % time limit  
    In.Res(1).tClim = NLSA.climatologyPeriod; % climatology time limits 

    trendStr = ''; % string identifier for detrening of target data

    % Source data specification 
    In.Src(1).field = 'sstw';      % area weighted SST
    In.Src(1).xLim  = [28 290];  % longitude limits
    In.Src(1).yLim  = [-60  20]; % latitude limits

    % Delay-embedding/finite-difference parameters; in-sample data
    In.Src(1).idxE      = 1 : NLSA.embWindow; % delay-embedding indices 
    In.Src(1).nXB       = 2;          % samples before main interval
    In.Src(1).nXA       = 2;          % samples after main interval
    In.Src(1).fdOrder   = 4;          % finite-difference order 
    In.Src(1).fdType    = 'central';  % finite-difference type
    In.Src(1).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res(1).nB    = 1; % partition batches
    In.Res(1).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = NLSA.kernel; % local distance
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
    In.nPhi       = 451;        % diffusion eigenfunctions to compute
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
    In.koopmanEpsilon = 5.0E-4;      % regularization parameter
    In.koopmanRegType = 'inv';     % regularization type
    In.idxPhiKoopman  = 1 : 401;   % diffusion eigenfunctions used as basis
    In.nPhiKoopman    = numel(In.idxPhiKoopman);        % Koopman eigenfunctions to compute
    In.nKoopmanPrj    = In.nPhiKoopman; % Koopman eigenfunctions for projection
    In.idxKoopmanRec = {[2 3] ...                   % annual
                         [7 8] ...                   % ENSO
                         [11 12 16 17] ...           % ENSO combination
                         [20 21] ...                 % 3-year mode
                         [7 8 11 12 16 17] ...       % ENSO + combination
                         [7 8 20 21] ...             % ENSO + 3-year
                         [7 8 11 12 16 17 20 21] ... % ENSO + combi. + 3-year
                       };


otherwise
        error('Invalid experiment')
end

%% PREPARE TARGET COMPONENTS (COMMON TO ALL MODELS)
%
% climStr is a string identifier for the climatology period relative to which
% anomalies are computed. 
%
% nETrg is the delay-embedding window for the target data

climStr = ['_' In.Res(1).tClim{1} '-' In.Res(1).tClim{2}];
nETrg   = 1; 

% Nino 3.4 index
In.Trg(1).field = ['sstmawav' climStr]; % (area-weighted SST anomaly)
In.Trg(1).xLim  = [190 240];            % longitude limits
In.Trg(1).yLim  = [-5 5];               % latitude limits


% Abbreviated target component names
In.targetComponentName   = ['nino_sst_ssh_air_prec_uv'];
In.targetRealizationName = '_';

% Prepare dalay-embedding parameters for target data
for iCT = 1 : numel(In.Trg)
        
    In.Trg(iCT).idxE      = 1 : nETrg;  % delay embedding indices 
    In.Trg(iCT).nXB       = 1;          % before main interval
    In.Trg(iCT).nXA       = 0;          % samples after main interval
    In.Trg(iCT).fdOrder   = 0;          % finite-difference order 
    In.Trg(iCT).fdType    = 'backward'; % finite-difference type
    In.Trg(iCT).embFormat = 'overlap';  % storage format
end


%% SERIAL DATE NUMBERS FOR IN-SAMPLE DATA
% Loop over the in-sample realizations
for iR = 1 : numel(In.Res)
    limNum = datenum(In.Res(iR).tLim, In.tFormat);
    nS = months(limNum(1), limNum(2)) + 1; 
    In.Res(iR).tNum = datemnth(limNum(1), 0 : nS - 1); 
end

%% CONSTRUCT NLSA MODEL
[model, In] = climateNLSAModel(In);
