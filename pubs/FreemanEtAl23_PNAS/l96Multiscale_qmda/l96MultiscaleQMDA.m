%% QUANTUM MECHANICAL DATA ASSIMILATION OF THE LORENZ 96 MULTISCALE MODEL
%
% Observed data are the slow variables x_1, ..., x_K.
% Predicted data is x_1. 
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
% The script provides options to run two cases: 
%   - A test case with small training and verification data. This will not lead 
%     to high-quality output, but is useful for testing the code.
%   - A case with larger training and verification data that will reproduce the
%     results in Figs. 2 and 3 of Freeman et al. The training and test data
%     specifications are as in Burov et al. (2020), "Kernel analog forecasting:
%     multiscale test problems", Multiscale Model. Simul. (2021), 19(2), 
%     1011-1040, https://doi.org/10.1137/20M1338289. This case results in the 
%     creation of large arrays as the data assimilation portion is vectorized 
%     over initial conditions and forecast lead times. To decrease memory 
%     footprint (at the expense of longer run times) increase the nBatch 
%     parameter in the kernel autotuning step (which controls the data chunk 
%     sizes used for kernel tuning) and/or create a new case with a smaller 
%     test dataset size. As presently configured, the case took several hours to
%     to run on a Linux server equipped with AMD EPYC 7713 processors. 

% Modified 2023/01/19
    

%% SCRIPT EXECUTION OPTIONS
% Data generation
ifTrainingData     = true;   % training data
ifTestData         = true;   % test data
ifTrainingTestData = false;  % generate training and test data in parallel  

% NLSA
ifNLSA    = true; % perform NLSA (source data, training phase)
ifNLSAObs = true; % perform NLSA (observed data, training phase)
ifNLSAOut = true; % perform out-of-sample extension (data assimilation phase)

% Data assimilation
ifDATrainingData  = true; % read training data for data assimilation
ifDATestData      = true; % read test data for data assimilation
ifKoopmanOp       = true; % compute Koopman operators
ifObservableOp    = true; % compute quantum mechanical observable operators  
ifAutotune        = true; % tune the observation kernel
ifFeatureOp       = true; % compute quantum mechanical feature operators
ifDA              = true; % perform data assimilation
ifDAErr           = true; % compute forecast errors

% IO options
ifSaveData        = true;  % save DA output to disk
ifLoadData        = false; % load DA output from disk (for plotting)
ifSaveOperators   = true;  % save DA operators (e.g., for use by Python code) 
ifPrintFig        = true;  % print figures to file

% Plotting options
ifPlotPrb       = true; % running probability forecast
ifPlotErr       = true; % plot forecast error
ifPlotErrMulti  = false; % plot forecast errors from multiple experiments
ifShowObs       = false; % show observations in plots


%% DATA & NLSA PARAMETERS
% Default parameters (changes may lead to inconsistent NLSA model)
NLSA.nX        = 9;     % Number of slow variables
NLSA.nY        = 8;     % Number of fast variables per slow variable
NLSA.hX        = -0.8;  % Coupling parameter from slow to fast variables
NLSA.hY        = 1;     % Coupling parameter from fast to slow variables    
NLSA.nSSpin    = 10000; % Number of spinup samples
NLSA.x0        = 1;     % Initial conditions for slow variables (training data)
NLSA.y0        = 1;     % Initial conditions for fast variables (training data)
NLSA.x0Out     = 1.2;   % Initial conditions for slow variables (test data)
NLSA.y0Out     = 1.2;   % Initial conditions for fast variables (test data)
NLSA.absTol    = 1E-7;  % Absolute tolerance for ODE solver
NLSA.relTol    = 1E-5;  % Relative tolerance for ODE solver

% User-adjustable parameters (safe to change) 
% Chaotic regime -- short dataset for testing
NLSA.F         = 10;            % Forcing parameter
NLSA.epsilon   = 1 / 128;       % Timesscale parameter for fast variables
NLSA.dt        = 0.05;          % Sampling interval
NLSA.nS        = 1000;          % Number of training samples
NLSA.nSOut     = 100;           % Number of test samples 
NLSA.nSSpin    = 100;           % Number of spinup samples
NLSA.idxXSrc   = 1 : NLSA.nX;   % State vector components to use for training
NLSA.idxXObs   = 1 : NLSA.nX;   % State vector components for observations
NLSA.embWindow = 1;             % Delay embedding window
NLSA.kernel    = 'l2';          % Kernel type
NLSA.ifDen     = true;          % Set true to use variable-bandwidth kernel 

% Chaotic regime -- dataset as in Burov et al. (2021) MMS. 
% NLSA.F         = 10;            % Forcing parameter
% NLSA.epsilon   = 1 / 128;       % Timesscale parameter for fast variables
% NLSA.dt        = 0.05;          % Sampling interval
% NLSA.nS        = 40000;         % Number of training samples
% NLSA.nSOut     = 7000;          % Number of test samples 
% NLSA.idxXSrc   = 1 : NLSA.nX;   % State vector components to use for training
% NLSA.idxXObs   = 1 : NLSA.nX;   % State vector components for observations
% NLSA.embWindow = 1;             % Delay embedding window
% NLSA.kernel    = 'l2';          % Kernel type
% NLSA.ifDen     = true;          % Set true to use variable-bandwidth kernel 

experiment   = experimentStr(NLSA);          % String identifier
TrainingPars = getTrainingDataPars(NLSA);    % Training data parameters
TestPars     = getTestDataPars(NLSA);        % Test data parameters
dataFunc     = @l96MultiscaleData;           % Data generation function 
modelFunc    = @l96MultiscaleQMDA_nlsaModel; % NLSA model function


%% SETUP QMDA PARAMETERS 
switch experiment

% Chaotic regime -- small dataset for testing
case 'F10.0_eps0.00781_dt0.05_nS1000_nSOut100_idxXSrc1-+1-9_idxXObs1-+1-9_emb1_l2_den'
    idxY  = 1; % predicted components 
    idxR  = 1; % realization (ensemble member) in assimilation phase   

    % QMDA parameters
    QMDA.nL          = 100; % eigenfunctions used for operator approximation
    QMDA.nQ          = 11; % quantization levels 
    QMDA.nTO         = 1; % timesteps between obs
    QMDA.nTF         = 50; % number of forecast timesteps (must be at least nTO)
    QMDA.epsilonScl  = 1;   % bandwidth scaling factor 
    QMDA.shape_fun   = @bump; % kernel shape function
    QMDA.ifVB        = true;  % use variable-bandwidth kernel 
    QMDA.ifSqrtm     = false; % apply square root to matrix-valued feature map
    
    % Number of parallel workers
    NPar.emb          = 0; % delay embedding
    NPar.nN           = 0; % nearest neighbor computation
    NPar.koopmanOp    = 0; % Koopman operator calculation 
    NPar.observableOp = 0; % Observable operator calculation 
    NPar.featureOp    = 0; % Feature operator calculation
    NPar.da           = 0; % Main QMDA loop

    % Plotting parameters
    Plt.tLim = [0 4]; % verification time interval to plot
    Plt.idxTF = [0 : 25 : 50] + 1; % forecast lead times
    Plt.idxY = 1; % estimated components for running probability forecast
    Plt.yLim = [-7 12]; % y axis limit for plots

    % Operator saving parameters
    Sav.idxU = [1 : 5]; % Koopman operators to save
    Sav.idxK = [1 : 5]; % Feature operators to save

% Chaotic regime -- dataset as in Burov et al. (2020) MMS. Reproduces Figs. 2 
% and 3 of Freeman et al. (2023) PNAS. 
case 'F10.0_eps0.00781_dt0.05_nS40000_nSOut7000_idxXSrc1-+1-9_idxXObs1-+1-9_emb1_l2_den'
    idxY  = 1; % predicted components 
    idxR  = 1;     % realization (ensemble member) in assimilation phase   

    % QMDA parameters
    QMDA.nL         = 2000; % eigenfunctions used for operator approximation
    QMDA.nQ         = 31; % quantization levels 
    QMDA.nTO        = 1; % timesteps between obs
    QMDA.nTF        = 300; % number of forecast timesteps (must be at least nTO)
    QMDA.epsilonScl = 0.6; % bandwidth scaling factor 
    QMDA.shape_fun  = @bump; % kernel shape function
    QMDA.ifVB       = true; % use variable-bandwidth kernel 
    QMDA.ifSqrtm    = false; % apply square root to matrix-valued feature map
    
    % Number of parallel workers
    NPar.emb          = 0; % delay embedding
    NPar.nN           = 0; % nearest neighbor computation
    NPar.koopmanOp    = 0; % Koopman operator calculation 
    NPar.observableOp = 0; % Observable operator calculation 
    NPar.featureOp    = 0; % Feature operator calculation
    NPar.da           = 0; % Main QMDA loop

    % Plotting parameters
    Plt.tLim = [65 85]; % verification time interval to plot
    Plt.idxTF = [0 : 25 : 100] + 1; % forecast lead times
    Plt.idxY = 1; % estimated components for running probability forecast
    Plt.yLim = [-7 12]; % y axis limit for plots

    % Operator saving parameters
    Sav.idxU = [1 : 100]; % Koopman operators to save
    Sav.idxK = [1 : 200]; % Feature operators to save

otherwise
    error('Invalid experiment')
end

q_experiment = qmdaStr(QMDA);

disp(['EXPERIMENT: ' experiment])
disp(['QMDA parameters: ' q_experiment])


%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes


%% PARALLEL POOL
% Create parallel pool if running NLSA and the NLSA model has been set up
% with parallel workers. We also create a parallel pool with 2 workers if
% the ifTrainingTestData option has been selected.
nPar = 0;
if ifTrainingTestData
    nPar = max(nPar, 2);
end
if ifNLSA
    nPar = max([nPar NPar.emb NPar.nN]);
end
if ifKoopmanOp
    nPar = max(nPar, NPar.koopmanOp);
end
if ifObservableOp
    nPar = max(nPar, NPar.observableOp);
end
if ifFeatureOp
    nPar = max(nPar, NPar.featureOp);
end
if ifDA
    nPar = max(nPar, NPar.da);
end
if nPar > 0
    poolObj = gcp('nocreate');
    if isempty(poolObj)
        poolObj = parpool(nPar);
    end
end


%% GENERATE TRAINING DATA
if ifTrainingData
    disp(['Generating training data using function ' ...
            func2str(dataFunc) '...'])
    t = tic;
    dataFunc(TrainingPars);
    toc(t)
end


%% GENERATE TEST DATA
if ifTestData
    disp(['Generating test data using function ' ...
            func2str(dataFunc) '...'])
    t = tic;
    dataFunc(TestPars);
    toc(t)
end

%% GENERATE TRAINING AND TEST DATA IN PARALLEL
if ifTrainingTestData
    disp(['Generating training and test data in parallel using function ' ...
            func2str(dataFunc) '...'])
    t = tic;
    parfor(i = 1 : 2, 2)
        if i == 1
            dataFunc(TrainingPars);
        else
            dataFunc(TestPars);
        end
    end
    toc(t)
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
% nSOut is the number of verification samples. 
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

[model, In, modelObs, InObs, OutObs] = modelFunc(experiment);

nY    = numel(idxY);
nSE   = getNTotalEmbSample(model); 
nSB   = getNXB(model.embComponent);
nSOut = getNTotalOutSample(modelObs);
nSO   = getNTotalOutEmbSample(modelObs);
nTO   = QMDA.nTO;
nTF   = QMDA.nTF;
nL    = QMDA.nL;
nQ    = QMDA.nQ;

nE           = floor((getEmbeddingWindow(model.embComponent) - 1) / 2);
idxT1        = getOrigin(model.trgEmbComponent(1)); 
idxT1Obs     = getOrigin(modelObs.trgEmbComponent(1)); 
nShiftTakens = idxT1 - nE - idxT1Obs;

if nShiftTakens < 0
    error(['Negative Takens delay embedding window shift. ' ...
           'Increase the delay embedding origin of the source data, ' ...
           'or decerase the delay embedding origin of the observed data.'])
end

idxTObs = 1 : nTO : nSO;
nDA     = numel(idxTObs); 

targetVar = {modelObs.trgComponent.tagC};
for iVar = 1 : numel(targetVar)
    targetVar{iVar} = ['x' targetVar{iVar}(5 : end)];
end

% Initialization/verification timestamps
iStart = idxT1Obs;
iEnd   = idxT1Obs + nSO - 1;
tOut   = (0 : nSOut - 1) * In.dt; % timestamps for true signal
tOut   = tOut(iStart : iEnd); % shift by delay window 
tObs   = (tOut(1) + (idxTObs - 1) * In.dt)'; % initialization times  
tLead  = (0 : nTF) * In.dt; % lead times 
tVer   = repmat(tObs, [1 nTF + 1]); % verification times
tVer   = (tVer + repmat(tLead, [nDA, 1]))';
tF = 0 : nTF; % forecast lead times


%% OUTPUT DIRECTORY
outDir = fullfile(pwd, 'figs', experiment, q_experiment);
if (ifPrintFig || ifSaveData || ifSaveOperators) && ~isdir(outDir)
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

        disp('Distance normalization for kernel density estimation...');
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

    disp('Kernel bandwidth tuning...'); t = tic;
    computeKernelDoubleSum(model)
    toc(t)

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

        disp('Distance normalization for kernel density steimation...');
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
    disp('Retrieving data assimilation training data...'); t = tic;
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
    y    = getData(modelObs.trgEmbComponent, [], [], idxY); 
    yL2 = std(y, 0, 2);

    % Align phiObs and y with the center of the embedding window
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
    disp('Retrieving data assimilation test data...'); t = tic;
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

    toc(t)
end
    
%% KOOPMAN OPERATORS
if ifKoopmanOp
    tic
    disp(sprintf('Computing Koopman operators for %i timesteps...', nTF))
    t = tic;
    U = koopmanOperator(1 : nTF, phi(:, 1 : nL), mu, nPar=NPar.koopmanOp);
    toc(t)

    if ifSaveOperators
        t  = tic;
        disp('Saving Koopman operators...')
        U_sav = U(:, :, Sav.idxU);
        dataFile = fullfile(outDir, 'u.mat');
        outVars = {'NLSA' 'QMDA' 'Sav' 'U_sav'};  
        save(dataFile, outVars{:}, '-v7.3')
        toc(t)
    end
end


%% MULTIPLICATION OPERATORS FOR TARGET OBSERVABLES
if ifObservableOp
    tic
    str = sprintf(['Computing multiplication operators for ' ...
                     '%i observables...'], nY);  
    disp(str)
    t = tic;

    % Compute multiplication operators representing y and y^2
    M  = multiplicationOperator(y', phi(:, 1 : nL), mu, ...
                                 NPar.observableOp);
    M2 = M ^ 2;

    if nQ > 0 
        b    = linspace(0, 1, nQ + 1); % CDF bins
        yQ = zeros(nQ + 1, nY);
        specIdx = zeros(nQ, 2, nY); % to rename idxQ
        MEval = zeros(nL, nY);
        MEvec = zeros(nL, nL, nY);

        parfor(iY = 1 : nY, NPar.observableOp)

            % Obtain quantization batch limits as uniform partition of the 
            % empirical inverse CDF of observably iY.
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

    if ifSaveOperators
        t  = tic;
        disp('Saving multiplication operators...')
        dataFile = fullfile(outDir, 'm.mat');
        outVars = {'NLSA' 'QMDA' 'M', 'M2'};  
        if nQ > 0
            outVars = [outVars {'yQ' 'dyQ' 'MEval' 'MEvec'}];
        end
        save(dataFile, outVars{:}, '-v7.3')
        toc(t)
    end
end


%% KERNEL TUNING
if ifAutotune
    disp('Computing pairwise distances for kernel tuning...')
    t = tic;
    if QMDA.ifVB
        d2 = rho .\ dmat(x) ./ rho';
    else
        d2 = dmat(x);
    end
    toc(t)

    disp('Performing kernel tuning...')
    t = tic ;
    [epsilonOpt, epsilonInfo] = autotune(d2(:), QMDA.shape_fun, ...
                                         exponent=[-40 40], nBatch=20, n=500);  
    toc(t)
end


%% FEATURE OPERATORS (QUANTUM EFFECT)
if ifFeatureOp

    str = sprintf(['Computing feature vectors for ' ...
                     '%i out-of-sample states...'], nSO);  
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
    K = multiplicationOperator(k, phi(:, 1 : nL), mu, NPar.featureOp);
    toc(t)

    if ifSaveOperators
        t  = tic;
        disp('Saving feature operators...')
        K_sav = K(:, :, Sav.idxK);
        dataFile = fullfile(outDir, 'k.mat');
        outVars = {'NLSA' 'QMDA' 'Sav' 'K_sav'};  
        save(dataFile, outVars{:}, '-v7.3')
        toc(t)
    end
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
                                     nTO, NPar.da);
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
        yDen = yPrb ./ dyQ;           % probability density 
    end

    if ifSaveData
        dataFile = fullfile(outDir, 'forecast.mat');
        outVars = {'NLSA' 'QMDA' 'tVer' 'tObs' 'tOut' 'yOut' ...
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
    disp('Computing forecast errors...')
    t = tic;
    if ifSaveOperators
        [~, yRMSE, yPC, yRef] = forecastError(yOut, yExp, nTO);
        yRef = yRef(:, :, Sav.idxK);
    else
        [~, yRMSE, yPC] = forecastError(yOut, yExp, nTO);
    end
    yRMSE = yRMSE ./ yL2;
    yRMSE_est = sqrt(mean(yStd .^ 2, 3)) ./ yL2;
    toc(t)

    if ifSaveData
        dataFile = fullfile(outDir, 'forecast_error.mat');
        save(dataFile, 'NLSA', 'QMDA', 'tF', 'yRMSE', 'yRMSE_est', 'yPC')
    end
    if ifSaveOperators
        dataFile = fullfile(outDir, 'y.mat');
        save(dataFile, 'NLSA', 'QMDA', 'Sav', 'yRef', '-v7.3')
    end
end


%% PLOT RUNNING PROBABILITY FORECAST 
if ifPlotPrb

    if ifLoadData
        dataFile = fullfile(outDir, 'forecast.mat');
        disp(sprintf('Loading forecast data from file %s...', ...
              dataFile))
        load(dataFile, 'tVer', 'tObs', 'tOut', 'yOut', 'yExp', ...
                        'yStd', 'yQ', 'yDen') 
    end

    nTPlt = numel(Plt.idxTF); % number of lead times to plot
    nYPlt = numel(Plt.idxY); % number of target variables to plot

    [idxTLimPlt, ~] = find(tObs == Plt.tLim, 2);
    idxTPlt = idxTLimPlt(1) : idxTLimPlt(2);
    
    % Figure parameters
    clear Fig
    Fig.nTileX     = 1;
    Fig.nTileY     = nTPlt;
    Fig.units      = 'inches';
    Fig.figWidth   = 9;    
    Fig.deltaX     = .55;
    Fig.deltaX2    = .9;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .3;
    Fig.gapX       = .2;
    Fig.gapY       = .6;
    Fig.gapT       = .25;
    Fig.aspectR    = (3 / 4) ^ 5;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [0.01 0];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    % Loop over target variables
    for iY = 1 : nYPlt

        % Set up figure and axes
        [fig, ax, axTitle] = tileAxes(Fig);

        % Loop over lead times
        for iT = 1 : nTPlt 

            iTF = (1 : nTO) + Plt.idxTF(iT) - 1;

            % Serial date numbers for verification times
            tPlt      = tVer(iTF, idxTPlt);
            tPlt      = tPlt(:)';
            ifTruePlt = tOut >= tPlt(1) & tOut <= tPlt(end);
            ifObsPlt  = tObs >= tPlt(1) & tObs <= tPlt(end); 
            %tLabels   = cellstr(datestr(tPlt , 'mm/yy')); 
        
            % Form grid of forecast times and bin centers. We add "ghost" 
            % points in the end to ensure that everything gets plotted using
            % pcolor.
            tGrd = [tPlt tPlt(end) + In.dt];
            yGrd = yQ(:, iY);
            yGrd(1) = min(y(iY, :));
            yGrd(end) = max(y(iY, :));

            % Assemble plot data for predicted mean and probability distribution
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
            hTrue = plot(tOut(ifTruePlt), ...
                          yOut(Plt.idxY(iY), find(ifTruePlt)), ...
                          'r-', 'lineWidth', 2);

            % Plot observations
            if ifShowObs
                hObs = plot(tObs(ifObsPlt), ...
                             yObs(Plt.idxY(iY), find(ifObsPlt)), ...
                             'r*', 'linewidth', 2);
            end

            set(gca, 'xLimSpec', 'tight')
            ylim(Plt.yLim)
            set(gca, 'cLim', [-3 0])
            axPos = get(gca, 'position');
            hC = colorbar('location', 'eastOutside');
            cPos = get(hC, 'position');
            cPos(3) = .5 * cPos(3);
            cPos(1) = cPos(1) + .08;
            set(gca, 'position', axPos)
            set(hC, 'position', cPos)
            ylabel(hC, 'log_{10} \rho')
            tl = get(gca, 'xLim');
            tl0 = round(tl(1));
            if isodd(tl0) 
                tl0 = tl0 - 1;
            end
            set(gca, 'xTick', tl0 : 2 : tl(2)) 
            ylabel('x_1')
            title(sprintf('Lead time \\tau = %2.2f', tLead(Plt.idxTF(iT))))
            if iT == nTPlt
                xlabel('verification time (model time units)')
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

        title(axTitle, 'Running forecasts: Lorenz 96 multiscale')

        % Print figure
        if ifPrintFig
            figFile = sprintf('figProb_%s.png', targetVar{Plt.idxY(iY)});
            figFile = fullfile(outDir, figFile);
            print(fig, figFile, '-dpng', '-r300') 
        end
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

    nYPlt = numel(Plt.idxY); % number of target variables to plot

    % Set up figure and axes 
    clear Fig
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

    % Plot skill scores
    [fig, ax, axTitle] = tileAxes(Fig);
    for iY = 1 : nY 

        % Normalized RMSE
        set(gcf, 'currentAxes', ax(1, iY))
        plot(tLead, yRMSE(iY, :), 'lineWidth', 1.5)
        plot(tLead, yRMSE_est(iY, :), 'r-', 'lineWidth', 1.5)

        grid on
        ylim([0 1.1])
        xlim([tLead(1) tLead(end)])
        set(gca, 'yTick', 0 : .2 : 1.2)  
        % ylabel('Normalized RMSE')
        legend('Normalized RMSE', 'Forecast standard deviation', ...
                'location', 'southEast')
        if iY == nY
            xlabel('Lead time \tau (model time units)')
        else
            set(gca, 'xTickLabel', [])
        end
        text(0.32, 1.03, '(a)')
        
        % Anomaly correlation
        set(gcf, 'currentAxes', ax(2, iY))
        plot(tLead, yPC(iY, :), 'lineWidth', 1.5)

        grid on
        ylim([0 1.1])
        xlim([tLead(1) tLead(end)])
        set(gca, 'yTick', 0 : .2 : 1.2, 'yAxisLocation', 'right')   
        % ylabel('Anomaly correlation')
        legend('Anomaly correlation', 'location', 'southWest')
        if iY == nY
            xlabel('Lead time \tau (model time units)')
        else
            set(gca, 'xTickLabel', [])
        end
        text(13.7, 1.03, '(b)')

        title(axTitle, 'Forecast skill: Lorenz 96 multiscale')

        % Print figure
        if ifPrintFig
            figFile = sprintf('figErr_%s.png', targetVar{Plt.idxY(iY)});
            figFile = fullfile(outDir, figFile);
            print(fig, figFile, '-dpng', '-r300') 
        end
    end
end

%% PLOT ERROR SCORES FROM MULTIPLE EXPERIMENTS
if ifPlotErrMulti
    % Plots Fig. S1 in the SI Appendix of Freeman et al. (2023). This requires
    % QMDA is run for the Hilbert space dimensions in the vector nLs below. 

    nLs = [512 1000 2000];
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
            plot(tLead, yRMSEs(iY, :, iExp), 'lineWidth', 1.5)
            legendStr{iExp} = sprintf('L = %i', nLs(iExp));
        end
        grid on
        ylim([0 1.1])
        xlim([tLead(1) tLead(end)])
        set(gca, 'yTick', 0 : .2 : 1.2)  
        ylabel('Normalized RMSE')
        legend(legendStr{:}, 'location', 'southEast')
        if iY == nY
            xlabel('Lead time \tau (model time units)')
        else
            set(gca, 'xTickLabel', [])
        end
        text(0.32, 1.03, '(a)')
            
        % Anomaly correlation
        set(gcf, 'currentAxes', ax(2, iY))
        for iExp = 1 : nExp
            plot(tLead, yPCs(iY, :, iExp), 'lineWidth', 1.5)
        end
        grid on
        ylim([0 1.1])
        xlim([tLead(1) tLead(end)])
        set(gca, 'yTick', 0 : .2 : 1.2, 'yAxisLocation', 'right')   
        ylabel('Anomaly correlation')
        if iY == nY
            xlabel('Lead time \tau (model time units)')
        else
            set(gca, 'xTickLabel', [])
        end
        text(13.7, 1.03, '(b)')
    end
    title(axTitle, 'Forecast skill: Lorenz 96 multiscale')

    % Print figure
    if ifPrintFig
        figFile = sprintf('figErrMulti_%s.png', targetVar{Plt.idxY});
        figFile = fullfile(outDir, figFile);
        print(fig, figFile, '-dpng', '-r300') 
    end
end


%% HELPER FUNCTIONS

% String identifier for data analysis experiment
function s = experimentStr(P)
    if ~iscell(P.idxXSrc)
        idxXSrc = {P.idxXSrc};
    else
        idxXSrc = P.idxXSrc;
    end
    nC = numel(idxXSrc);
    str = cell(1, nC);
    for iC = 1 : nC
        str{iC} = idx2str(idxXSrc{iC}, 'idxXSrc');
    end
    s = strjoin_e({sprintf('F%1.1f', P.F )  ...
                     sprintf('eps%1.3g', P.epsilon) ...
                     sprintf('dt%1.3g', P.dt) ...
                     sprintf('nS%i', P.nS) ...
                     sprintf('nSOut%i', P.nSOut) ...
                     str{:} ...
                     idx2str(P.idxXObs, 'idxXObs') ...
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
end

% Parameters for training data generation
function D = getTrainingDataPars(P) 
    D.Pars.nX       = P.nX;
    D.Pars.nY       = P.nY;
    D.Pars.F        = P.F;
    D.Pars.hX       = P.hX;
    D.Pars.hY       = P.hY;
    D.Pars.epsilon  = P.epsilon;
    D.Time.nS       = P.nS;
    D.Time.nSSpin   = P.nSSpin;
    D.Time.dt       = P.dt;
    D.Ode.x0        = P.x0;
    D.Ode.y0        = P.y0;
    D.Ode.absTol    = P.absTol;
    D.Ode.relTol    = P.relTol;
    D.Opts.ifCenter = false;
    D.Opts.ifWrite  = true;
    D.Opts.idxX     = cell(1, P.nX); 

    for iX = 1 : P.nX
        D.Opts.idxX{iX} = iX;
    end
end

% Parameters for test data generation
function D = getTestDataPars(P) 
    D.Pars.nX       = P.nX;
    D.Pars.nY       = P.nY;
    D.Pars.F        = P.F;
    D.Pars.hX       = P.hX;
    D.Pars.hY       = P.hY;
    D.Pars.epsilon  = P.epsilon;
    D.Time.nS       = P.nSOut;
    D.Time.nSSpin   = P.nSSpin;
    D.Time.dt       = P.dt;
    D.Ode.x0        = P.x0Out;
    D.Ode.y0        = P.y0Out;
    D.Ode.absTol    = P.absTol; 
    D.Ode.relTol    = P.relTol;
    D.Opts.ifCenter = false;
    D.Opts.ifWrite  = true;
    D.Opts.idxX     = cell(1, P.nX); 

    for iX = 1 : P.nX
        D.Opts.idxX{iX} = iX;
    end
end
