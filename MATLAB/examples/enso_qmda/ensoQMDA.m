%% QUANTUM MECHANICAL DATA ASSIMILATION OF THE EL NINO SOUTHERN OSCILLATION
%
% Observed data is Indo-Pacific SST.
% Predicted data is Nino 3.4 index.
%
% Output from each step of the deta generation and NLSA parts is saved on disk
% in directory 'data'. These parts of the script need only be run once unless
% changes to the dataset and/or NLSA parameters are made. 
%
% Output fom the data assimilation (DA) portion of the script is saved in 
% directory 'figs' if any of the variables ifSaveData, ifSaveOperators, or 
% ifPrintFig are set to true. Setting ifLoadData to true then allows using the
% DA output (e.g., for plotting figures) without re-running the DA code. 
%
% Modified 2023/01/19
    
%% SCRIPT EXECUTION OPTIONS
% Data extraction from netCDF files
ifDataSrc    = true; % source data (training phase)
ifDataTrg    = true; % target data (training phase)
ifDataObs    = true; % observed data (training phase)
ifDataOutObs = true; % observed data (data assimilation phase) 
ifDataOutTrg = true; % target data (data assimilation phase)

% NLSA
ifNLSA    = true; % perform NLSA (source data, training phase)
ifNLSAObs = true; % perform NLSA (observed data, training phase)
ifNLSAOut = true; % perform out-of-sample extension (data assimilation phase)

% Data assimilation
ifDATrainingData  = true; % read training data for data assimilation
ifDATestData      = true; % read test data for data assimilation
ifKoopmanOp       = true; % compute Koopman operators
ifObservableOp    = true; % compute quantum mechanical observable operators  
ifAutotune        = true;  % tune the observation kernel
ifFeatureOp       = true; % compute quantum mechanical feature operators
ifDA              = true; % perform data assimilation
ifDAErr           = true; % compute forecast errors

% IO options
ifSaveData        = true;  % save DA output to disk
ifLoadData        = false;  % load data from disk
ifPrintFig        = false; % print figures to file

% Plotting/movie options
ifPlotPrb    = true; % running probability forecast
ifPlotErr   = true; % plot forecast errors
ifPlotErrMulti  = false; % plot forecast errors from multiple experiments
ifShowObs   = false; % show observations in plots


%% NLSA PARAMETERS
% ERSSTv4 reanalysis
% NLSA.dataset           = 'ersstV4';
%NLSA.trainingPeriod    = {'197001' '202002'};
%NLSA.climatologyPeriod = {'198101' '201012'};
%NLSA.sourceVar         = {'IPSST'};
%NLSA.targetVar         = {'Nino3.4'};
%NLSA.embWindow         = 48; 
%NLSA.kernel            = 'cone';
%NLSA.ifDen             = false;

% CCSM4 control run
NLSA.dataset           = 'ccsm4Ctrl';
NLSA.trainingPeriod    = {'000101' '109912'};
NLSA.testPeriod        = {'110001' '130012'};
NLSA.climatologyPeriod = NLSA.trainingPeriod;
% NLSA.sourceVar         = {'IPSST'};
% NLSA.sourceVar         = {'Nino3.4'};
NLSA.sourceVar         = {'IPSST' 'Nino3.4'};
NLSA.obsVar            = {'IPSST'};
NLSA.targetVar         = {'Nino3.4' 'Nino4' 'Nino3' 'Nino1+2'};
% NLSA.embWindow         = 9;
NLSA.embWindow         = 11; 
% NLSA.embWindow         = 13; 
% NLSA.embWindow         = 17; 
% NLSA.embWindow         = 23; 
% NLSA.embWindow         = 12; 
NLSA.kernel            = 'l2';
NLSA.ifDen             = true;
% NLSA.kernel            = 'cone';
% NLSA.ifDen             = false;

% CCSM4 control run -- small datasaet for testing
NLSA.dataset           = 'ccsm4Ctrl';
NLSA.trainingPeriod    = {'000101' '003012'};
NLSA.testPeriod        = {'120001' '121212'};
NLSA.climatologyPeriod = NLSA.trainingPeriod;
NLSA.sourceVar         = {'IPSST' 'Nino3.4'};
NLSA.targetVar         = {'Nino3.4' 'Nino4' 'Nino3' 'Nino1+2'};
NLSA.embWindow         = 3; 
NLSA.kernel            = 'l2';
NLSA.ifDen             = true;

experiment = experimentStr(NLSA);

%% SETUP GLOBAL PARAMETERS 
switch experiment

case 'ccsm4Ctrl_000101-109912_110001-130012_IPSST_IPSST_emb11_l2_den'

    idxY  = [1]; % predicted  components (Nino 3.4 index) 
    idxR  = 1;     % realization (ensemble member) in assimilation phase   

    % QMDA parameters
    QMDA.nL         = 1500;  % eigenfunctions used for operator approximation
    % QMDA.nL         = 3000;  % eigenfunctions used for operator approximation
    QMDA.nQ         = 31;     % number of spectral bins
    QMDA.nTO        = 1;     % timesteps between obs
    QMDA.nTF        = 12;    % number of forecast timesteps (must be at least nTO)
    QMDA.epsilonScl = 1.03; % bandwidth scaling factor 
    QMDA.shape_fun  = @bump; % kernel shape function
    % QMDA.shape_fun  = @rbf;  % kernel shape function
    QMDA.ifVB       = true; % use variable-bandwidth kernel  
    QMDA.ifSqrtm    = false; % apply square root to matrix-valued feature map

    % Number of parallel workers
    NPar.koopmanOp    = 0; % Koopman operator calculation  
    NPar.observableOp = 0; % Observable operator calculation 
    NPar.featureOp    = 0; % Feature operator calculation
    NPar.qmda         = 0; % Main QMDA loop

    % Plotting parameters
    Plt.tLim = {'120005' '122005'}; % time limit to plot
    Plt.idxTF = [0 : 3 : 12] + 1; % lead times for running forecast
    Plt.idxY = 1; % estimated components for running probability forecast
    Plt.yQLim = [-3 3];

case 'ccsm4Ctrl_000101-109912_110001-130012_Nino3.4_IPSST_emb11_l2_den'
    idxY  = [1]; % predicted  components (Nino 3.4 index) 
    idxR  = 1;     % realization (ensemble member) in assimilation phase   

    % QMDA parameters
    % QMDA.nL         = 1500;  % eigenfunctions used for operator approximation
    QMDA.nL         = 3000;  % eigenfunctions used for operator approximation
    QMDA.nQ         = 31;     % number of spectral bins
    QMDA.nTO        = 1;     % timesteps between obs
    QMDA.nTF        = 12;    % number of forecast timesteps (must be at least nTO)
    QMDA.epsilonScl = 1.03; % bandwidth scaling factor 
    QMDA.shape_fun  = @bump; % kernel shape function
    % QMDA.shape_fun  = @rbf;  % kernel shape function
    QMDA.ifVB       = true; % use variable-bandwidth kernel  
    QMDA.ifSqrtm    = false; % apply square root to matrix-valued feature map

    % Number of parallel workers
    NPar.koopmanOp    = 0; % Koopman operator calculation  
    NPar.observableOp = 0; % Observable operator calculation 
    NPar.featureOp    = 0; % Feature operator calculation
    NPar.qmda         = 0; % Main QMDA loop

    % Plotting parameters
    Plt.tLim = {'120005' '122005'}; % time limit to plot
    Plt.idxTF = [0 : 3 : 12] + 1; % lead times for running forecast
    Plt.idxY = 1; % estimated components for running probability forecast
    Plt.yQLim = [-3 3];


case 'ccsm4Ctrl_000101-109912_110001-130012_IPSST+Nino3.4_IPSST_emb11_l2_den'
    % Cased used in Freeman et al. (2023), PNAS.

    idxY  = [1]; % predicted  components (Nino 3.4 index) 
    idxR  = 1;     % realization (ensemble member) in assimilation phase   

    % QMDA parameters
    % QMDA.nL         = 500;  % eigenfunctions used for operator approximation
    % QMDA.nL         = 1000;  % eigenfunctions used for operator approximation
    QMDA.nL         = 1500;  % eigenfunctions used for operator approximation
    % QMDA.nL         = 2000;  % eigenfunctions used for operator approximation
    QMDA.nQ         = 31;     % number of spectral bins
    QMDA.nTO        = 1;     % timesteps between obs
    QMDA.nTF        = 12;    % number of forecast timesteps (must be at least nTO)
    % QMDA.epsilonScl = 1.03; % bandwidth scaling factor 
    QMDA.epsilonScl = 1.0; % bandwidth scaling factor 
    % QMDA.epsilonScl = 0.97; % bandwidth scaling factor 
    % QMDA.epsilonScl = 0.93; % bandwidth scaling factor 
    QMDA.shape_fun  = @bump; % kernel shape function
    % QMDA.shape_fun  = @rbf;  % kernel shape function
    QMDA.ifVB       = true; % use variable-bandwidth kernel  
    QMDA.ifSqrtm    = false; % apply square root to matrix-valued feature map
    QMDA.ifPolar    = false; % perform polar decomposition of Koopman 

    % Number of parallel workers
    NPar.koopmanOp    = 0; % Koopman operator calculation  
    NPar.observableOp = 0; % Observable operator calculation 
    NPar.featureOp    = 0; % Feature operator calculation
    NPar.qmda         = 0; % Main QMDA loop

    % Plotting parameters
    Plt.tLim = {'120005' '122005'}; % time limit to plot
    Plt.idxTF = [0 : 3 : 12] + 1; % lead times for running forecast
    Plt.idxY = 1; % estimated components for running probability forecast
    Plt.yQLim = [-3 3];

case 'ccsm4Ctrl_000101-003012_120001-121212_IPSST+Nino3.4_IPSST_emb3_l2_den'
    % Small dataset for testing

    idxY  = [1]; % predicted  components (Nino 3.4 index) 
    idxR  = 1;     % realization (ensemble member) in assimilation phase   

    % QMDA parameters
    QMDA.nL         = 100;  % eigenfunctions used for operator approximation
    QMDA.nQ         = 11;     % number of spectral bins
    QMDA.nTO        = 1;     % timesteps between obs
    QMDA.nTF        = 12;    % number of forecast timesteps (must be at least nTO)
    QMDA.epsilonScl = 1.0; % bandwidth scaling factor 
    QMDA.shape_fun  = @bump; % kernel shape function
    QMDA.ifVB       = true; % use variable-bandwidth kernel  
    QMDA.ifSqrtm    = false; % apply square root to matrix-valued feature map
    QMDA.ifPolar    = false; % perform polar decomposition of Koopman 

    % Number of parallel workers
    NPar.koopmanOp    = 0; % Koopman operator calculation  
    NPar.observableOp = 0; % Observable operator calculation 
    NPar.featureOp    = 0; % Feature operator calculation
    NPar.qmda         = 0; % Main QMDA loop

    % Plotting parameters
    Plt.tLim = {'120005' '121005'}; % time limit to plot
    Plt.idxTF = [0 : 3 : 12] + 1; % lead times for running forecast
    Plt.idxY = 1; % estimated components for running probability forecast
    Plt.yQLim = [-3 3];

case 'ccsm4Ctrl_000101-109912_110001-130012_IPSST+Nino3.4_IPSST+Nino3.4_emb11_l2_den'

    idxY  = [1]; % predicted  components (Nino 3.4 index) 
    idxR  = 1;     % realization (ensemble member) in assimilation phase   

    % QMDA parameters
    QMDA.nL         = 1000;  % eigenfunctions used for operator approximation
    % QMDA.nL         = 1500;  % eigenfunctions used for operator approximation
    % QMDA.nL         = 2000;  % eigenfunctions used for operator approximation
    % QMDA.nL         = 3000;  % eigenfunctions used for operator approximation
    QMDA.nQ         = 31;     % number of spectral bins
    QMDA.nTO        = 1;     % timesteps between obs
    QMDA.nTF        = 12;    % number of forecast timesteps (must be at least nTO)
    % QMDA.epsilonScl = 1.03; % bandwidth scaling factor 
    QMDA.epsilonScl = 1;
    QMDA.shape_fun  = @bump; % kernel shape function
    % QMDA.shape_fun  = @rbf;  % kernel shape function
    QMDA.ifVB       = true; % use variable-bandwidth kernel  
    QMDA.ifSqrtm    = false; % apply square root to matrix-valued feature map

    % Number of parallel workers
    NPar.koopmanOp    = 0; % Koopman operator calculation  
    NPar.observableOp = 0; % Observable operator calculation 
    NPar.featureOp    = 0; % Feature operator calculation
    NPar.qmda         = 0; % Main QMDA loop

    % Plotting parameters
    Plt.tLim = {'120005' '122005'}; % time limit to plot
    Plt.idxTF = [0 : 3 : 12] + 1; % lead times for running forecast
    Plt.idxY = 1; % estimated components for running probability forecast
    Plt.yQLim = [-3 3];

otherwise
    error(['Invalid experiment '  experiment])
end

q_experiment = qmdaStr(QMDA);

disp(['EXPERIMENT: ' experiment])
disp(['QMDA parameters: ' q_experiment])

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% EXTRACT SOURCE DATA
if ifDataSrc
    for iVar = 1 : numel(NLSA.sourceVar)
        msgStr = sprintf('Reading source data %s...', ...
                          NLSA.sourceVar{iVar});
        disp(msgStr) 
        t = tic;
        importData(NLSA.dataset, NLSA.trainingPeriod, ...
                    NLSA.sourceVar{iVar}, NLSA.climatologyPeriod) 
        toc(t)
    end
end

%% EXTRACT TARGET DATA
if ifDataTrg
    for iVar = 1 : numel(NLSA.targetVar)
        msgStr = sprintf('Reading target data %s...', ...
                           NLSA.targetVar{iVar});
        disp(msgStr) 
        t = tic;
        importData(NLSA.dataset, NLSA.trainingPeriod, ...
                    NLSA.targetVar{iVar}, NLSA.climatologyPeriod) 
        toc(t)
    end
end

%% EXTRACT OBSERVED DATA
if ifDataObs
    for iVar = 1 : numel(NLSA.obsVar)
        msgStr = sprintf('Reading observed data %s...', ...
                          NLSA.obsVar{iVar});
        disp(msgStr) 
        t = tic;
        importData(NLSA.dataset, NLSA.trainingPeriod, ...
                    NLSA.obsVar{iVar}, NLSA.climatologyPeriod) 
        toc(t)
    end
end

%% EXTRACT OUT-OF-SAMPLE OBSERVED DATA
if ifDataOutObs
    for iVar = 1 : numel(NLSA.obsVar)
        msgStr = sprintf('Reading out-of-sample observed data %s...', ...
                          NLSA.obsVar{iVar});
        disp(msgStr) 
        t = tic;
        importData(NLSA.dataset, NLSA.testPeriod, ...
                    NLSA.obsVar{iVar}, NLSA.climatologyPeriod) 
        toc(t)
    end
end

%% EXTRACT OUT-OF-SAMPLE TARGET DATA
if ifDataOutTrg
    for iVar = 1 : numel(NLSA.targetVar)
        msgStr = sprintf('Reading out-of-sample target data %s...', ...
                           NLSA.targetVar{iVar});
        disp(msgStr) 
        t = tic;
        importData(NLSA.dataset, NLSA.testPeriod, ...
                    NLSA.targetVar{iVar}, NLSA.climatologyPeriod) 
        toc(t)
    end
end

%% BUILD NLSA MODELS, DETERMINE BASIC ARRAY SIZES
% In and InObs are data structures containing the NLSA parameters for the 
% source and observed data, respectively, in the training phase. OutObs is a
% data structure containing the NLSA parameters for the observed data in the 
% data assimilation (out-of-sample) phase.
%
% nY is the number of response variables (Nino indices).
%
% nSE is the number of training samples available after Takens delay
% embedding.
%
% nSO is the number of verification samples available after Takens delay
% embedding. 
%
% nE is equal to half of the delay embedding window length employed in the 
% training data. 
%
% idxT1 is the initial time stamp in the training data ("origin") where delay 
% embedding is performed.
%
% idxT1Obs is the initial time stamp in the observed data ("origin") where delay
% embedding is performed.
%
% nShiftTakens is the temporal shift applied to align the data from the 
% observations NLSA model (modelObs) with the center of the Takens embedding 
% window eployed in the source model (model).  We compute nShiftTakens using
% nE, idxT1 and idxTObs.
%
% idxTObs are the time indices in the test phase at which observations are
% made.
%
% nDA is the number of data assimilation cycles, i.e., number of elements of
% idxTObs.

[model, In, modelObs, InObs, OutObs] = ensoQMDA_nlsaModel(experiment);

nY  = numel(idxY);
nSE = getNTotalEmbSample(model);    
nSB = getNXB(model.embComponent);
nSO = getNTotalOutEmbSample(modelObs);
nTO = QMDA.nTO;
nTF = QMDA.nTF;
nL  = QMDA.nL;
nQ  = QMDA.nQ;

nE           = floor((getEmbeddingWindow(model.embComponent) - 1) / 2);
idxT1        = getOrigin(model.trgEmbComponent(1)); 
idxT1Obs     = getOrigin(modelObs.trgEmbComponent(1)); 
nShiftTakens = idxT1 - nE - idxT1Obs;
% nShiftTakens = 8; % Temporary change for testing
% nShiftTakens = 0; % Temporary change for testing

if nShiftTakens < 0
    error(['Negative Takens delay embedding window shift. ' ...
             'Increase the delay embedding origin of the source data, ' ...
             'or decerase the delay embedding origin of the observed data.'])
end

idxTObs = 1 : nTO : nSO;
nDA     = numel(idxTObs); 


%% PARALLEL POOL
% Create parallel pool if running NLSA and the NLSA model has been set up
% with parallel workers. This part can be commented out if no parts of the
% NLSA code utilizing parallel workers are being executed. 
%
% In.nParE is the number of parallel workers for delay-embedded distances
% In.nParNN is the number of parallel workers for nearest neighbor search
if ifNLSA
    if isfield(In, 'nParE') && In.nParE > 0
        nPar = In.nParE;
    else
        nPar = 0;
    end
    if isfield(In, 'nParNN') && In.nParNN > 0
        nPar = max(nPar, In.nParNN);
    end
    if nPar > 0
        poolObj = gcp('nocreate');
        if isempty(poolObj)
            poolObj = parpool(nPar);
        end
    end
end


%% OUTPUT DIRECTORY
outDir = fullfile(pwd, 'figs', experiment, q_experiment);
if (ifPrintFig || ifSaveData) && ~isdir(outDir)
    mkdir(outDir)
end

%% PERFORM NLSA FOR SOURCE DATA
% Output from each step is saved on disk.
if ifNLSA

    disp('Takens delay embedding for source data...'); t = tic; 
    computeDelayEmbedding(model)
    toc(t)

    % The following step is needed only if we are using velocity-dependent
    % kernels.
    if isa(model.embComponent, 'nlsaEmbeddedComponent_xi')
      disp('Phase space velocity (time tendency of data)...'); t = tic; 
      computeVelocity(model)
      toc(t)
    end

    % The following steps are needed only if we are using variable-bandwidth
    % kernels.
    if isa(model, 'nlsaModel_den')
        fprintf('Pairwise distances for density data, %i/%i...\n', ...
                  iProc, nProc); 
       t = tic;
       computeDenPairwiseDistances(model, iProc, nProc)
       toc(t)

       disp('Distance normalization for kernel density steimation...');
       t = tic;
       computeDenBandwidthNormalization(model);
       toc(t)

       disp('Kernel bandwidth tuning for density estimation...'); t = tic;
       computeDenKernelDoubleSum(model);
       toc(t)

       disp('Kernel density estimation...'); t = tic;
       computeDensity(model);
       toc(t)

       disp('Takens delay embedding for density data...'); t = tic;
       computeDensityDelayEmbedding(model);
       toc(t)
    end

    fprintf('Pairwise distances (%i/%i)...\n', iProc, nProc); t = tic;
    computePairwiseDistances(model, iProc, nProc)
    toc(t)

    disp('Distance symmetrization...'); t = tic;
    symmetrizeDistances(model)
    toc(t)

    if isa(model.diffOp, 'nlsaDiffusionOperator_gl_mb') ...
            || isa(model.diffOp, 'nlsaDiffusionOperator_gl_mb_bs')
        disp('Kernel bandwidth tuning...'); t = tic;
        computeKernelDoubleSum(model)
        toc(t)
    end

    disp('Kernel eigenfunctions...'); t = tic;
    computeDiffusionEigenfunctions(model)
    toc(t)
end

%% PERFORM NLSA FOR OBSERVED DATA
if ifNLSAObs

    % Execute NLSA steps. Output from each step is saved on disk.

    disp('Takens delay embedding for observed data...'); t = tic; 
    computeDelayEmbedding(modelObs)
    toc(t)

    disp('Takens delay embedding for target data...'); t = tic;
    computeTrgDelayEmbedding(modelObs)
    toc(t)

    % The following step is needed only if we are using velocity-dependent
    % kernels.
    if isa(modelObs.embComponent, 'nlsaEmbeddedComponent_xi')
        disp('Phase space velocity (time tendency of data)...'); t = tic; 
        computeVelocity(modelObs)
        toc(t)
    end

    % The following steps are needed only if we are using variable-bandwidth
    % kernels.
    if isa(modelObs, 'nlsaModel_den')
        fprintf('Pairwise distances for density data, %i/%i...\n', ...
                  iProc, nProc); 
        t = tic;
        computeDenPairwiseDistances(modelObs, iProc, nProc)
        toc(t)

        disp('Distance normalization for kernel density estimation...');
        t = tic;
        computeDenBandwidthNormalization(modelObs);
        toc(t)

        disp('Kernel bandwidth tuning for density estimation...'); t = tic;
        computeDenKernelDoubleSum(modelObs);
        toc(t)

        disp('Kernel density estimation...'); t = tic;
        computeDensity(modelObs);
        toc(t)
    end
end

%% DO OUT-OF-SAMPLE EXTENSION FOR OBSERVED DATA
if ifNLSAOut
    
    disp('Takens delay embedding for out-of-sample observed data...'); 
    t = tic;
    computeOutDelayEmbedding(modelObs)
    toc(t)

    disp('Takens delay embedding for out-of-sample target data...'); t = tic;
    computeOutTrgDelayEmbedding(modelObs)
    toc(t)
    
    % The following step is needed only if we are using velocity-dependent
    % kernels.
    if isa(modelObs.outEmbComponent, 'nlsaEmbeddedComponent_xi')
        disp('Phase space velocity for out-of-sample data...'); t = tic; 
        computeOutVelocity(modelObs)
        toc(t)
    end

    % The following steps are needed only if we are using variable-bandwidth
    % kernels.
    if isa(modelObs, 'nlsaModel_den')
        fprintf('OSE pairwise distances for density data... %i/%i\n', ...
                  iProc, nProc); 
        t = tic;
        computeOseDenPairwiseDistances(modelObs, iProc, nProc)
        toc(t)

        disp('OSE density bandwidth normalization...'); t = tic;
        computeOseDenBandwidthNormalization(modelObs)
        toc(t)

        disp('OSE kernel density estimation...'); t = tic;
        computeOseDensity(modelObs)
        toc(t)
    end
end


%% READ TRAINING DATA FOR DATA ASSIMILATION
if ifDATrainingData
    disp('Retrieving data assimilation training data...'); 
    t = tic;
    
    % Eigenfunctions from source data 
    [phi, mu, lambda] = getDiffusionEigenfunctions(model); 

    % Observed data
    x = getData(modelObs.embComponent);

    % Read kernel density if using variable bandwidth kernel
    if QMDA.ifVB
        [~, rhoInfo] = computeOptimalBandwidth(modelObs.density);
        rho = getDensity(modelObs.density) .^ (- 1 / rhoInfo.dEst);
    end

    % Target data. yL2 is the empirical L2 norm. 
    y = getData(modelObs.trgEmbComponent, [], [], idxY); 
    yL2 = std(y, 0, 2);

    % Align x, y, and rho with the center of the embedding window
    iStart = nShiftTakens + 1;
    iEnd   = iStart + nSE - 1;
    x      = x(:, iStart : iEnd);
    y      = y(:, iStart : iEnd);
    if QMDA.ifVB
        rho = rho(iStart : iEnd);
    end
    toc(t)
end

%% READ TEST DATA FOR DATA ASSIMILATION
if ifDATestData
    disp('Retrieving data assimilation test data...'); 
    t = tic;

    xOut = getData(modelObs.outEmbComponent);
    xObs = xOut(:, idxTObs);
    if QMDA.ifVB
        rhoOut = getDensity(modelObs.oseDensity) .^ (- 1 / rhoInfo.dEst);
        rhoObs = rhoOut(idxTObs);
    end

    % Test (out-of-sample) target data
    yOut = getData(modelObs.outTrgEmbComponent, [], [], idxY); 
    yObs = yOut(:, idxTObs);

    % Initialization/verification timestamps
    iStart  = idxT1Obs;
    iEnd    = idxT1Obs + nSO - 1;
    tNumOut = getOutTime(modelObs); % serial time numbers for true signal
    tNumOut = tNumOut(iStart : iEnd); % shift by delay window 
    tNumObs = datemnth(tNumOut(1), idxTObs - 1)'; % initialization times  
    tNumVer = repmat(tNumObs, [1 nTF + 1]); % verification times
    tNumVer = datemnth(tNumVer, repmat(0 : nTF, [nDA, 1]))';

    % Forecast lead times
    tF = 0 : nTF; 

    toc(t)
end
    
%% KOOPMAN OPERATORS
if ifKoopmanOp
    disp(sprintf('Computing Koopman operators for %i timesteps...', nTF))
    t = tic;
    U = koopmanOperator(1 : nTF, phi(:, 1 : nL), mu, ...
                        nPar=NPar.koopmanOp, polar=QMDA.ifPolar);
    toc(t)
end

%% MULTIPLICATION OPERATORS FOR TARGET OBSERVABLES
if ifObservableOp
    str = sprintf(['Computing multiplication operators for ' ...
                     '%i observables...'], nY);  
    disp(str)
    t = tic;

    % Compute multiplication operators representing y and y^2
    M  = multiplicationOperator(y', phi(:, 1 : nL), mu, ...
                                 NPar.observableOp);
    M2 = M ^ 2;
    
    % Compute bins for evaluation of the spectral measure of M
    if nQ > 0 
        b    = linspace(0, 1, nQ + 1); % CDF bins
        yQ = zeros(nQ + 1, nY);
        specIdx = zeros(nQ, 2, nY); % to rename idxQ
        MEval = zeros(nL, nY);
        MEvec = zeros(nL, nL, nY);
        parfor(iY = 1 : nY, NPar.observableOp)

            % Obtain quantization batch limits as uniform partition of the 
            % empirical inverse CDF of observable iY.
            yQ(:, iY) = ksdensity(y(iY, :), b, 'Function', 'icdf');

            % Compute the spectrum of the multiplication operator representing
            % observable iY.
            [evecM, evalM] = eig(M(:, :, iY), 'vector');
            [MEval(:, iY), idx] = sort(evalM, 'ascend'); 
            MEvec(:, :, iY) = evecM(:, idx, iY);

            % Partition spectrum into batches
            inds = discretize(MEval(:, iY), yQ(:, iY)); 
            specIdxLoc = zeros(nQ, 2);
            for iQ = 1 : nQ
                specIdxLoc(iQ, 1) = find(inds == iQ, 1, 'first');  
                specIdxLoc(iQ, 2) = find(inds == iQ, 1, 'last');  
            end
            specIdx(:, :, iY) = specIdxLoc;
        end

        yQ(1, :)   = min(y');
        yQ(end, :) = max(y'); 

        dyQ = yQ(2 : end, :) - yQ(1 : end - 1, :);
    end
    toc(t)
end

%% KERNEL TUNING
if ifAutotune
    disp('Computing pairwise distances for kernel tuning...')
    t = tic;
    d2 = dmat(x);
    if QMDA.ifVB
        d2 = rho .\ dmat(x) ./ rho';
    end
    toc(t)

    disp('Performing kernel tuning...')
    t = tic ;
    [epsilonOpt, epsilonInfo] = autotune(d2(:), QMDA.shape_fun, ...
                                         exponent=[-10 10], nBatch=5, n=500);  
    toc(t)
end


%% FEATURE OPERATORS (QUANTUM EFFECT)
if ifFeatureOp

    str = sprintf(['Computing feature vectors for ' ...
                     '%i out-of-sample states...'], nDA);  
    disp(str)
    t = tic;
    if QMDA.ifVB
        if QMDA.ifSqrtm
            k = QMDA.shape_fun(...
                rho .\ dmat(x, xObs) ./ rhoObs' ...
                / (epsilonOpt * QMDA.epsilonScl) ^ 2) ;
        else
            k = sqrt(...
                QMDA.shape_fun(...
                rho .\ dmat(x, xObs) ./ rhoObs' ...
                / (epsilonOpt * QMDA.epsilonScl) ^ 2));
        end
    else
        if QMDA.ifSqrtm
            k = QMDA.shape_fun(...
                dmat(x, xObs) ...
                / (epsilonOpt * QMDA.epsilonScl) ^ 2);
        else 
            k = sqrt(...
                QMDA.shape_fun(...
                dmat(x, xObs) ...
                / (epsilonOpt * QMDA.epsilonScl) ^ 2));
        end
    end
    if any(all(k == 0, 1))
        error('Zero feature vectors detected.')
    end
    toc(t)

    disp('Computing feature operators...')
    t = tic;
    if QMDA.ifSqrtm
        K = sqrtMultiplicationOperator(k, phi(:, 1 : nL), mu, ...
                                        NPar.featureOp);
    else
        K = multiplicationOperator(k, phi(:, 1 : nL), mu, NPar.featureOp);
    end
    toc(t)
end

%% DATA ASSIMILATION
if ifDA 
    % xi0 is the initial wavefunction. We set xi0 to the constant wavefunction 
    % representing the invariant distrubution of the system, i.e., only the 
    % leading spectral expansion coefficient 
    xi0      = zeros(1, nL);
    xi0(1) = 1;

    if nQ > 0 
        [yExp, yStd, yPrb] = qmda(K, U, M, M2, MEvec, specIdx, xi0, ...
                                     nTO, NPar.qmda);
    else
        [yExp, yStd] = qmda(K, U, M, M2, [], [], xi0, nTO, NPar.qmda);
    end
    yExp0 = yExp(:, :, 1);
    yStd0 = yExp(:, :, 1);
    yExp = yExp(:, :, 2 : end);
    yStd = yStd(:, :, 2 : end);
    if nQ > 0
        yPrb0 = yPrb(:, :, :, 1); % probability mass
        yPrb = yPrb(:, :, :, 2 : end); % probability mass
        yDen0 = yPrb0 ./ dyQ; 
        yDen = yPrb ./ dyQ;              % probability density 
    end

    if ifSaveData
        dataFile = fullfile(outDir, 'forecast.mat');
        outVars = {'NLSA' 'QMDA' 'tNumVer' 'tNumObs' 'tNumOut' 'yOut' ...
                    'yExp' 'yStd' 'yExp0' 'yStd0'};  
        if nQ > 0
            outVars = [outVars {'yQ' 'yPrb' 'yDen' 'yPrb0' 'yDen0'}];
        end
        save(dataFile, outVars{:}, '-v7.3')
    end
end

%% DATA ASSIMILATION ERRORS
% yRMSE is an array of size [nY, nTF + 1] such that yRMSE(i, j) is the 
% root mean square error of the mean forecast yExp for target variable 
% idxY(i) at lead time j - 1. 
%
% yPC is as yRMSE but stores the anomaly correlation coefficient between the
% true signal and mean forecast.
%
% yRMSE_est is the estimated forecast error based on the predicted standard 
% deviation yStd.
if ifDAErr
    disp('Computing forecast skill scores...'); 
    t = tic;
    [~, yRMSE, yPC] = forecastError(yOut, yExp);
    yRMSE = yRMSE ./ yL2;
    yRMSE_est = sqrt(mean(yStd .^ 2, 3)) ./ yL2;
    toc(t)

    if ifSaveData
        dataFile = fullfile(outDir, 'forecast_error.mat');
        save(dataFile, 'NLSA', 'QMDA', 'tF', 'yRMSE', 'yRMSE_est', 'yPC')
    end
end


%% PLOT RUNNING PROBABILITY FORECAST 
if ifPlotPrb

    if ifLoadData
        dataFile = fullfile(outDir, 'forecast.mat');
        disp(sprintf('Loading forecast data from file %s...', ...
              dataFile))
        load(dataFile, 'tNumVer', 'tNumObs', 'tNumOut', 'yOut', 'yExp', ...
                        'yStd', 'yQ', 'yDen') 
    end

    nTPlt = numel(Plt.idxTF); % number of lead times to plot
    nYPlt = numel(Plt.idxY); % number of target variables to plot
    nTickSkip = 12;           % number of months between time axis ticks

    % Determine time indices to plot within verification time interval
    tNumLimPlt = datenum(Plt.tLim, 'yyyymm');
    [idxTLimPlt, ~] = find(tNumObs == tNumLimPlt', 2);
    idxTPlt = idxTLimPlt(1) : idxTLimPlt(2);


    % Figure parameters
    Fig.nTileX     = nYPlt;
    Fig.nTileY     = nTPlt;
    Fig.units      = 'inches';
    Fig.figWidth   = 9;    
    Fig.deltaX     = .55;
    Fig.deltaX2    = .9;
    Fig.deltaY     = .75;
    Fig.deltaY2    = .25;
    Fig.gapX       = .2;
    Fig.gapY       = .7;
    Fig.gapT       = .25; 
    Fig.aspectR    = (3 / 4) ^ 5;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [0.007 0];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    % Set up figure and axes
    [fig, ax, axTitle] = tileAxes(Fig);

    % Loop over target variables
    for iY = 1 : nYPlt

        % Loop over lead times
        for iT = 1 : nTPlt 

            iTF = (1 : nTO) + Plt.idxTF(iT) - 1;

            % Serial date numbers and labels for verification times
            tPlt      = tNumVer(iTF, idxTPlt);
            tPlt      = tPlt(:)';
            ifTruePlt = tNumOut >= tPlt(1) & tNumOut <= tPlt(end);
            ifObsPlt  = tNumObs >= tPlt(1) & tNumObs <= tPlt(end); 
            tLabels   = cellstr(datestr(tPlt , 'mm/yy')); 
        
            % Form grid of forecast times and bin centers. We add "ghost" 
            % points in the end to ensure that everything gets plotted using
            % pcolor.
            tGrd = [tPlt tPlt(end) + 1];
            yGrd = yQ(:, iY);
            yGrd(1) = Plt.yQLim(1);
            yGrd(end) = Plt.yQLim(end);

            % Assemble plot data for mean and probability distribution
            yExpPlt = yExp(Plt.idxY(iY), iTF, idxTPlt);
            yExpPlt = yExpPlt(:);
            yDenPlt = yDen(:, Plt.idxY(iY), iTF, idxTPlt);
            yDenPlt = reshape(yDenPlt, nQ, numel(tPlt));
            yDenPlt = [yDenPlt; yDenPlt(end, :)];
            yDenPlt = [yDenPlt yDenPlt(:, end)];

            set(gcf, 'currentAxes', ax(iY, iT))

            % Plot predicted probability density
            hPrb = pcolor(tGrd, yGrd, log10(yDenPlt)); 
            set(hPrb, 'edgeColor', 'none')

            % Plot predicted mean 
            hExp = plot(tPlt, yExpPlt, 'k-', 'lineWidth', 2);

            % Plot true signal 
            hTrue = plot(tNumOut(ifTruePlt), ...
                          yOut(Plt.idxY(iY), find(ifTruePlt)), ...
                          'r-', 'lineWidth', 2);
            % Plot observations
            if ifShowObs
                plot(tNumObs(ifObsPlt), ...
                      yObs(Plt.idxY(iY), find(ifObsPlt)), ...
                      'r*', 'linewidth', 2)
            end

            set(gca, 'xTick', tPlt(1 : nTickSkip : end), ...
                      'xTickLabel', tLabels(1 : nTickSkip : end'), ...
                      'xLimSpec', 'tight')
            ylim([-2.8 3])
            set(gca, 'cLim', [-3 0])
            axPos = get(gca, 'position');
            hC = colorbar('location', 'eastOutside');
            cPos = get(hC, 'position');
            cPos(3) = .5 * cPos(3);
            cPos(1) = cPos(1) + .08;
            set(gca, 'position', axPos)
            set(hC, 'position', cPos)
            ylabel(hC, 'log_{10}\rho')
            
            ylabel('Nino 3.4')
            title(sprintf('Lead time \\tau = %i months', Plt.idxTF(iT) - 1))

            if iT == nTPlt
                xlabel('verification time')
            end

            % Add legend
            if iT == 1 
                if ifShowObs
                    hL = legend([hTrue hObs hExp], ...
                                  'true', 'observations', 'mean forecast', ...
                                  'location', 'northWest');
                else
                    hL = legend([hTrue hExp], ...
                                  'true', 'mean forecast', ...
                                  'location', 'northWest');
                end
                sL = hL.ItemTokenSize;
                sL(1) = .5 * sL(1);
                hL.ItemTokenSize = sL;
                lPos = get(hL, 'position');
                lPos(2) = lPos(2) + 0.04;
                set(hL, 'position', lPos)
            end
        end
    end

    title(axTitle, 'Running forecasts: CCSM4 ENSO')

    % Print figure
    if ifPrintFig
        figFile = fullfile(outDir, 'nino34_prob.png');
        print(fig, figFile, '-dpng', '-r300') 
    end
end

%% PLOT ERROR SCORES
if ifPlotErr

    if ifLoadData
        dataFile = fullfile(outDir, 'forecast_error.mat');
        disp(sprintf('Loading forecast error data from file %s...', ...
              dataFile))
        load(dataFile, 'tF', 'yRMSE', 'yRMSE_est', 'yPC')
    end

    % Set up figure and axes 
    Fig.nTileX     = 2;
    Fig.nTileY     = nY;
    Fig.units      = 'inches';
    Fig.figWidth   = 8; 
    Fig.deltaX     = .4;
    Fig.deltaX2    = .4;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .25;
    Fig.gapX       = .3;
    Fig.gapY       = .3;
    Fig.gapT       = 0.05; 
    Fig.aspectR    = 3 / 4;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [0.02 0];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [fig, ax, axTitle] = tileAxes(Fig);

    % Plot skill scores
    for iY = 1 : nY 

        % Normalized RMSE
        set(gcf, 'currentAxes', ax(1, iY))
        plot(tF, yRMSE(iY, :), 'linewidth', 1.5)
        plot(tF, yRMSE_est(iY, :), 'r-', 'lineWidth', 1.5)

        grid on
        ylim([0 1])
        xlim([tF(1) tF(end)])
        set(gca, 'xTick', tF(1) : 3 : tF(end), 'yTick', 0 : .2 : 1.2)  
        % ylabel('Normalized RMSE')
        legend('Normalized RMSE', 'Forecast standard deviation', ...
                'location', 'southEast')
        if iY == nY
            xlabel('Lead time \tau (months)')
        else
            set(gca, 'xTickLabel', [])
        end
        text(0.3, 0.95, '(a)')
        
        % Anomaly correlation
        set(gcf, 'currentAxes', ax(2, iY))
        plot(tF, yPC(iY, :), 'linewidth', 1.5)

        grid on
        ylim([0 1])
        xlim([tF(1) tF(end)])
        set(gca, 'xTick', tF(1) : 3 : tF(end), 'yTick', 0 : .2 : 1.2, ...
                  'yAxisLocation', 'right')   
        legend('Anomaly correlation', 'location', 'southWest')
        if iY == nY
            xlabel('Lead time \tau (months)')
        else
            set(gca, 'xTickLabel', [])
        end
        text(11, 0.95, '(b)')
    end

    % Add figure title
    title(axTitle, 'Forecast skill: CCSM4 ENSO')

    % Print figure
    if ifPrintFig
        figFile = fullfile(outDir, 'nino34_err.png');
        print(fig, figFile, '-dpng', '-r300') 
    end
end

%% PLOT ERROR SCORES FROM MULTIPLE EXPERIMENTS
if ifPlotErrMulti

    nLs = [500 1000 1500 2000];
    nExp = numel(nLs);

    % Load data to plot
    QMDAs = repmat(QMDA, 1, nExp);
    yRMSEs = zeros(nY, nTF + 1, nExp);
    yPCs = zeros(nY, nTF + 1, nExp);
    for iExp = 1 : nExp
        QMDAs(iExp).nL = nLs(iExp);
        outDirExp = fullfile(pwd, 'figs', experiment, qmdaStr(QMDAs(iExp)));
        dataFile = fullfile(outDirExp, 'forecast_error.mat');
        load(dataFile, 'tF', 'yRMSE', 'yRMSE_est', 'yPC')
        yRMSEs(:, :, iExp) = yRMSE;
        yPCs(:, :, iExp) = yPC;
    end

    % nYPlt = numel(Plt.idxY); % number of target variables to plot

    % Set up figure and axes 
    clear Fig
    Fig.nTileX     = 2;
    Fig.nTileY     = nY;
    Fig.units      = 'inches';
    Fig.figWidth   = 8; 
    Fig.deltaX     = .6;
    Fig.deltaX2    = .6;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .25;
    Fig.gapX       = .3;
    Fig.gapY       = .3;
    Fig.gapT       = 0.05; 
    Fig.aspectR    = 3 / 4;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [0.02 0];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    % Plot skill scores
    [fig, ax, axTitle] = tileAxes(Fig);
    legendStr = cell(1, nExp);
    for iY = 1 : nY 
        % Normalized RMSE
        set(gcf, 'currentAxes', ax(1, iY))
        for iExp = 1 : nExp
            plot(tF, yRMSEs(iY, :, iExp), 'lineWidth', 1.5)
            legendStr{iExp} = sprintf('L = %i', nLs(iExp));
        end
        grid on
        ylim([0 1])
        xlim([tF(1) tF(end)])
        set(gca, 'xTick', tF(1) : 3 : tF(end), 'yTick', 0 : .2 : 1.2)  
        ylabel('Normalized RMSE')
        legend(legendStr{:}, 'location', 'southEast')
        if iY == nY
            xlabel('Lead time \tau (model time units)')
        else
            set(gca, 'xTickLabel', [])
        end
        text(0.3, 0.95, '(a)')
            
        % Anomaly correlation
        set(gcf, 'currentAxes', ax(2, iY))
        for iExp = 1 : nExp
            plot(tF, yPCs(iY, :, iExp), 'lineWidth', 1.5)
        end
        grid on
        ylim([0 1])
        xlim([tF(1) tF(end)])
        set(gca, 'xTick', tF(1) : 3 : tF(end), 'yTick', 0 : .2 : 1.2, ...
                  'yAxisLocation', 'right')   
        ylabel('Anomaly correlation')
        if iY == nY
            xlabel('Lead time \tau (months)')
        else
            set(gca, 'xTickLabel', [])
        end
        text(11, 0.95, '(b)')
    end
    title(axTitle, 'Forecast skill: CCSM4 ENSO')

    % Print figure
    if ifPrintFig
        figFile = fullfile(outDir, 'nino34_errs.png');
        print(fig, figFile, '-dpng', '-r300') 
    end
end


%% HELPER FUNCTIONS

% String identifier for data analysis experiment
function s = experimentStr(P)
    s = strjoin_e({P.dataset ...
                     strjoin_e(P.trainingPeriod, '-') ... 
                     strjoin_e(P.testPeriod, '-') ...  
                     strjoin_e(P.sourceVar, '+') ...
                     strjoin_e(P.obsVar, '+') ...
                     sprintf('emb%i', P.embWindow) ...
                     P.kernel}, ...
                   '_');

    if isfield(P, 'ifDen')
        if P.ifDen
            s = [s '_den'];
        end
    end
end

% String identifier for QMDA parameters
function s = qmdaStr(P)
    s = strjoin_e({sprintf('nTO%i', P.nTO) ...
                     sprintf('nTF%i', P.nTF) ...
                     sprintf('nL%i', P.nL) ...
                     func2str(P.shape_fun) ...
                     sprintf('eps%1.2f', P.epsilonScl)}, ...
                   '_');
    if P.ifVB
        s = [s '_vb'];
    end
    if P.ifSqrtm
        s = [s '_sqrtm'];
    end
    if P.ifPolar
        s = [s '_polar'];
    end
end
