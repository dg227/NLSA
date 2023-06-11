 % CALCULATION OF KOOPMAN RMM INDICES
%
% The script calls the function specified by the function handle modelFunc to 
% create an object of nlsaModel class, which encodes all aspects and parameters 
% of the calculation, such as location of source data, kernel parameters, 
% Koopman operator approximation parameters, etc. 
%
% Results from each stage of the calculation are written on disk. Below is a
% summary of basic commands to access the code output:
%
% z     = getKoopmanEigenfunctions(model); % Koopman eigenfunctions
% gamma = getKoopmanEigenvalues(model);    % Koopman eigenvalues  
% T     = getKoopmanEigenperiods(model);   % Koopman eigenperiods
%
% Modified 2023/06/02


%% SCRIPT EXECUTION OPTIONS
% Data extraction and RMM index computations
ifImportData  = true; % import RMM data into appropriate format for NLSA
ifNLSA        = true; % perform NLSA
ifKoopmanEig  = true; % compute Koopman eigenfunctions
ifEOF         = true; % perform EOF analysis
ifRMM         = true; % compute EOF- and Koopman-based RMM indices
ifRMMTable    = true; % output Koopman RMM indices in ASCII form


%% DATA & NLSA PARAMETERS
NLSA.dataset   = 'LGPS23'; % data from Lintner et al. (2023) GRL 
NLSA.tFormat   = 'yyyymmdd'; % date format
NLSA.tStart    = '19980101'; % start time in dataset
NLSA.tLim      = {'19980101' '20191230'}; % analysis interval
NLSA.var       = 'RMM'; % input variable
NLSA.embWindow = 64; % delay embedding window (days)
NLSA.kernel    = 'cone'; % kernel type
NLSA.ifDen     = false; % set true to use variable-bandwidth kernel 

strFunc   = @experimentStr_rmm; % function to create string identifier
dataFunc  = @importData_rmm; % data import function
modelFunc = @nlsaModel_rmm; % function for building NLSA model

experiment = strFunc(NLSA); 
disp(['NLSA EXPERIMENT: ' experiment])


%% BATCH PROCESSING
% Not necessary to use nProc > 1 for small datasets such as meridionally 
% averaged RMM data.
iProc = 1; % current process
nProc = 1; % number of processes


%% OUTPUT DIRECTORY
outDir = fullfile(pwd, 'out', experiment);
if ifRMMTable && ~isdir(outDir)
    mkdir(outDir)
end


%% IMPORT DATA
if ifImportData
    disp(['Importing data using function ' func2str(dataFunc) '...'])
    t = tic;
    dataFunc(NLSA);
    toc(t)
end


%% BUILD NLSA MODEL
disp('Building NLSA model...')
t = tic;
model = modelFunc(NLSA);
nSE = getNTotalEmbSample(model);    
nSB = getNXB(model.embComponent);
nE  = getEmbeddingWindow(model.embComponent);
nShiftTakens = nSB + nE - 1;
toc(t)


%% PERFORM NLSA 
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

    % The following step is only needed if query partition is employed for  
    % source data.
    if ~isempty(model.embComponentQ)
        disp('Forming query partition for source data...'); t = tic; 
        computeDelayEmbeddingQ(model)
        toc(t)
    end

    % The following step is only needed if test partition is employed for source
    % data.
    if ~isempty(model.embComponentT)
        disp('Forming test partition for source data...'); t = tic; 
        computeDelayEmbeddingT(model)
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

        % The next step is only needed if a query partition was used for the 
        % density data.
        if ~isempty(model.denEmbComponentQ)
            disp('Density splitting...'); t = tic;
            computeDensitySplitting(model);
            toc(t)
        end

        disp('Takens delay embedding for density data...'); t = tic;
        computeDensityDelayEmbedding(model);
        toc(t)
        % The following step is only needed if query partition is employed for  
        % density data.
        if ~isempty(model.denEmbComponentQ)
            disp('Forming query partition for density data...'); t = tic; 
            computeDensityDelayEmbeddingQ(model)
            toc(t)
        end

        % The following step is only needed if test partition is employed for 
        % density data.
        if ~isempty(model.denEmbComponentT)
            disp('Forming test partition for density data...'); t = tic; 
            computeDensityDelayEmbeddingT(model)
            toc(t)
        end
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


%% COMPUTE KOOPMAN EIGENFUNCTIONS
if ifKoopmanEig
    disp('Koopman eigenfunctions...'); t = tic;
    computeKoopmanEigenfunctions(model, 'ifLeftEigenfunctions', true)
    toc(t)
end


%% PERFORM EOF ANALYSIS
if ifEOF
    disp('EOF analysis...'); t = tic;
    x = getData(model.srcComponent);
    a = double(x - mean(x, 2));
    [u, s, v] = svds(a);
    toc(t)
end


%% COMPUTE EOF AND KOOPMAN RMM INDICES
if ifRMM
    disp('RMM indices...'); t = tic;
    T = getKoopmanEigenperiods(model);
    idxMJO = find(abs(T - 50) < 5, 1);
    disp(sprintf(['Leading intraseasonal Koopman mode: ' ...
                  'index = %i, eigenperiod = %1.2f days'], idxMJO, T(idxMJO)))
    z = getKoopmanEigenfunctions(model);
    RMM.koopman = z(:, idxMJO);
    idxT = (1 : nSE) + nShiftTakens;
    eof = complex(v(idxT, 1), v(idxT, 2)) * sqrt(nSE);  
    corr_eof = [corr(eof, RMM.koopman) corr(conj(eof), RMM.koopman)];
    [amplCorr, idxCorr] = max(abs(corr_eof)); 
    if idxCorr == 1
        RMM.eof = eof;
    else
        RMM.eof = conj(eof);
    end
    RMM.koopman = conj(corr_eof(idxCorr)) / amplCorr * RMM.koopman; 
    tNum = getSrcTime(model);
    RMM.tNum = tNum(idxT);
    corr1 = corr(real(RMM.eof), real(RMM.koopman));
    corr2 = corr(imag(RMM.eof), imag(RMM.koopman));
    disp('Correlation coefficients between EOF and Koopman:')
    disp(sprintf('First component:  %1.2f', corr1))
    disp(sprintf('Second component: %1.2f', corr2))
    disp(sprintf('Amplitude:        %1.2f', amplCorr))


    %% Alternative rotation approach (yields similar results as the previous one)
    % thetas = linspace(0, 2 * pi, 100);
    % maxcorr = -Inf;
    % RMM.eof = eof;
    % for iTheta = 1 : 100
    %     corr_theta = corr(real(RMM.eof), ...
    %                       real(exp(i * thetas(iTheta)) * RMM.koopman));
    %     if corr_theta > maxcorr
    %         theta_rot = thetas(iTheta);        
    %         maxcorr = corr_theta;
    %     end
    % end
    % RMM.koopman = exp(i * theta_rot) * RMM.koopman;
end


if ifRMMTable
    tStart = datenum('19790101', 'yyyymmdd'); 
    tEnd = datenum('20191231', 'yyyymmdd'); 
    nT = tEnd - tStart + 1;

    tableFile = fullfile(outDir, 'koopman_rmm.txt');
    fid = fopen(tableFile, 'w');
    iCount = 1;
    for iT = 1 : nT
        t = tStart + iT - 1;
        if t >= RMM.tNum(1) && t <= RMM.tNum(end)
            rmm = RMM.koopman(iCount);
            rmm1 = real(RMM.koopman(iCount));
            rmm2 = imag(RMM.koopman(iCount));
            rmma = abs(RMM.koopman(iCount));
            tableRow = sprintf('%i\t%i\t%i\t0\t%1.3f\t%1.3f\t%1.3f', ...
                               year(t), month(t), day(t), rmm1, rmm2, rmma);
            iCount = iCount + 1;
        else
            tableRow = sprintf('%i\t%i\t%i\t0\tNaN\tNaN\tNaN', ...
                               year(t), month(t), day(t));
        end
        fprintf(fid, '%s \n', tableRow); 
    end
    fclose(fid);
end


%% PERFORM RECONSTRUCTION OF INPUT VARIABLES BASED ON KOOPMAN EIGENFUNCTIONS
% if ifKoopmanRec
%     disp('Takens delay embedding for target data...'); t = tic; 
%     computeTrgDelayEmbedding(model)
%     toc(t)

%     disp('Projection of target data onto kernel eigenfunctions...'); t = tic;
%     computeProjection(model)
%     toc(t)

%     disp('Projection of target data onto Koopman eigenfunctions...'); t = tic;
%     computeKoopmanProjection(model)
%     toc(t)
% end
