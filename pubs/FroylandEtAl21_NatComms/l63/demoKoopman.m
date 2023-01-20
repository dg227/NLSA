% DEMO OF KOOPMAN SPECTRAL ANALYSIS APPLIED TO LORENZ 63 (L63) DATA
%
% This script demonstrates the identification of coherent observables of the L63
% system using approximate eigenfunctions of the Koopman generator on the L2
% space of observables associated with the SRB measure of the system. 
%
% The generator is approximated through its matrix representation in a 
% data-driven basis for L2 consisting of eigenfunctions of a kernel integral 
% operator. This integral operator employs a variable-bandwidth Gaussian kernel
% acting on delay-coordinate-mapped data. The action of the generator on the 
% kernel eigenfunctions is approximated using finite differences in time. 
%
% The generator is regularized by adding a small multiple of a diffusion 
% operator which is diagonal in the kernel eigenfunction basis. The approximate
% generator eigenfunctions are ordered in increasing order of the corresponding
% Dirichlet energy.
%
% The following test case is provided:
%
% '16k_dt0.01_nEL800': 16000 samples, sampling interval 0.01, 800 delays 
%
% Additional test cases may be added in the function demoKoopman_nlsaModel. 
%
% Moreover, there is an option to run principal component analysis (PCA) on
% the data for comparison.
%
% The kernel employed is a variable-bandwidth Gaussian kernel, normalized to a
% symmetric Markov kernel. This requires a kernel density estimation step to
% compute the bandwidth functions, followed by a normalization step to form the
% kernel matrix. See the Methods section in [1] and the pseudocode in 
% references [2, 3] below for further details. 
%
% The script calls the function demoKoopman_nlsaModel to create an object of
% nlsaModel class, which encodes all aspects and parameters of the calculation,
% such as location of source data, kernel parameters, Koopman operator 
% approximation parameters, etc. 
%
% Results from each stage of the calculation are written on disk. Below is a
% summary of basic commands to access the code output:
%
% z     = getKoopmanEigenfunctions(model); -- Koopman eigenfunctions
% gamma = getKoopmanEigenvalues(model);    -- Koopman eigenvalues  
% T     = getKoopmanEigenperiods(model);   -- Koopman eigenperiods
%
% Setting the variables ifPlotEig and ifMovieEig to true produces a figure and
% a movie analogous to Fig. 2 and Movie 1 in [1], but using an approximation
% of the generator as opposed to the transfer operator used in [1].
%
% Figure/movie caption:
%
% Comparison of PCA (covariance) eigenfunctions (a-d) and generator 
% eigenfunctions (e-h) for extraction of approximately cyclic observables of 
% the Lorenz 63 (L63) chaotic system.} Panel (a) shows the principal component 
% (PC) corresponding to the leading empirical orthogonal function (EOF) as a 
% scatterplot (color is the EOF value) on the L63 attractor computed from a 
% dataset of 16,000 points along a single L63 trajectory, sampled at an interval% of Delta t = 0.01 time units. The black line shows a portion of the dynamical
% trajectory spannine 10 time units, corresponding to the time series shown in 
% Panels (d, h) and phase portraits in Panels (c, g). Panel (b) shows the phase
% angle on the attractor obtained by treating the leading two PCs as the real 
% and imaginary parts of a complex observable.
%
% The black line depicts the same portion of the dynamical trajectory as the 
% black line in Panel (a). Panel (c) shows a 2D projection associated with the 
% leading two EOF PCs for the same time interval as Panel (d). Since these PCs 
% correspond to linear projections of the data onto the corresponding EOFs, 
% the evolution in the 2D phase space spanned by PC_1 and PC_2 has comparable 
% complexity to the "raw" L63 dynamics, exhibiting a chaotic mixing of two 
% cycles associated with the two lobes of the attractor. 
%    
% Panels (e-h) show the corresponding results to Panels (a-d), respectively, 
% obtained from the leading non-constant eigenfunction g_1 of the generator 
% (see Methods in [1]). Panel (f) shows the argument of the complex-valued g_1 
% (color is the argument) evaluated at the 16,000 points in the trajectory. 
% Notice that there is a cyclic "rainbow" of color as one progresses around 
% each individual L63 attractor wing in phase space. Panel (g) plots these 
% same arguments of g_1, now in the complex plane, demonstrating that the 
% output of g_1 lies approximately on the unit circle. Panel~(h) shows the real
% part of the trajectory in Panel (g) plotted versus time, illustrating 
% approximately simple harmonic motion. Thus, the eigenfunction g_1 of the 
% generator extracts the dominant cyclic behavior of L63 on the attractor's 
% wings.
%
% References:
% 
% [1] G. Froyland, D. Giannakis, B. Lintner, M. Pike, J. Slawinska (2021). 
%     Spectral analysis of climate dynamics with operator-theoretic approaches. 
%     Nat. Commun. 12, 6570. https://doi.org/10.1038/s41467-021-26357-x.
%
% [2] D. Giannakis (2019). Data-driven spectral decomposition and forecasting
%     of ergodic dynamical systems. Appl. Comput. Harmon. Anal., 47, 338-396. 
%     https://dx.doi.org/10.1016/j.acha.2017.09.001.
%
% [3] S. Das, D. Giannakis, J. Slawinska (2021). Reproducing kernel Hilbert
%     space compactification of unitary evolution groups., Appl. Comput. 
%     Harmon. Anal. 54, 75-136. https://dx.doi.org/10.1016/j.acha.2021.02.004
% 
% Modified 2021/09/13

%% EXPERIMENT SPECIFICATION AND SCRIPT EXECUTION OPTIONS
experiment   = '16k_dt0.01_nEL800'; 

ifSourceData = true;  % generate source data
ifNLSA       = true;  % run NLSA (kernel eigenfunctions)
ifKoopman    = true;  % compute Koopman eigenfunctions 
ifPCA        = true;  % run PCA (for comparison with Koopman)
ifPlotEig    = true;  % plot Koopman and PCA eigenfunctions
ifMovieEig   = false; % make eigenfunction movie illustrating coherence
ifPrintFig   = true;  % print figures to file


%% GLOBAL PARAMETERS
% idxZ:       Eigenfunction to plot
% idxTLim:    Time interval to plot
% figDir:     Output directory for plots
% markerSize: For eigenfunction scatterplots
% signPC:     Sign multiplication factors for principal components

switch experiment

case '16k_dt0.01_nEL800'
    idxZ       = 2;     
    idxTLim    = [2601 3601]; % approx 10 Lyapunov times
    markerSize = 7;         
    signPC     = [-1 1];
    
otherwise
    error('Invalid experiment.')
end

% Figure/movie directory
figDir = fullfile(pwd, 'figs', experiment);
if ~isdir(figDir)
    mkdir(figDir)
end

%% EXTRACT SOURCE DATA
if ifSourceData
    disp('Generating source data...'); t = tic;
    demoKoopman_data(experiment) 
    toc(t)
end

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES
% In is a data structure containing the NLSA parameters for the training data.
%
% nSE is the number of samples avaiable for data analysis after Takens delay
% embedding.
%
% nSB is the number of samples left out in the start of the time interval (for
% temporal finite differnences employed in the kernel).
%
% nEL is the Takens embedding window length (in number of timesteps)
%
% nShiftTakens is the temporal shift applied to align eigenfunction data with 
% the center of the Takens embedding window. 

disp('Building NLSA model...'); t = tic;
[model, In] = demoKoopman_nlsaModel(experiment); 
toc(t)

nSE          = getNTotalSample(model.embComponent);
nSB          = getNXB(model.embComponent);
nEL          = getEmbeddingWindow(model.embComponent) - 1;
nShiftTakens = round(nEL / 2);


%% PERFORM NLSA
if ifNLSA
    
    % Execute NLSA steps. Output from each step is saved on disk.

    disp('Takens delay embedding...'); t = tic; 
    computeDelayEmbedding(model)
    toc(t)


    fprintf('Pairwise distances for density data, %i/%i...\n', iProc, nProc); 
    t = tic;
    computeDenPairwiseDistances(model, iProc, nProc)
    toc(t)

    disp('Distance normalization for kernel density estimation...'); t = tic;
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

%% COMPUTE EIGENFUNCTIONS OF KOOPMAN GENERATOR
if ifKoopman
    disp('Koopman eigenfunctions...'); t = tic;
    computeKoopmanEigenfunctions(model)
    toc(t)
end

%% PERFORM PCA
if ifPCA

    disp('Principal component analysis...'); t = tic;
    x = getData(model.srcComponent);
    [PCA.u, PCA.s, PCA.v] = svd(x - mean(x, 2), 'econ');
    PCA.v = PCA.v * sqrt(getNTotalSample(model.srcComponent));
    toc(t)
end

%% PLOT EIGENFUNCTIONS
if ifPlotEig
    
    % Retrieve source data and Koopman eigenfunctions
    x = getData(model.srcComponent);
    x = x(:, 1 + nShiftTakens : nSE + nShiftTakens);
    z = getKoopmanEigenfunctions(model);
    omega = getKoopmanEigenfrequencies(model);
    T = getKoopmanEigenperiods(model);

    % Determine amplitude and phase angle
    a = max(abs(z), [], 1);  % z amplitude
    angl = angle(z(:, idxZ)); % z phase

    % Construct 2D observable based on PCs
    zPC = PCA.v(1 + nShiftTakens : nSE + nShiftTakens, [1 2]) ...
        .* signPC / sqrt(2);
    anglPC = angle(complex(zPC(:, 1), zPC(:, 2))); % z phase
    aPC = max(abs(zPC), [], 1); % PC amplitude 

    % Determine number of temporal samples; assign timestamps
    nFrame = idxTLim(2) - idxTLim(1) + 1;
    t = (0 : nFrame - 1) * In.dt;  

    % Set up figure and axes 
    Fig.nTileX     = 4;
    Fig.nTileY     = 2;
    Fig.units      = 'inches';
    Fig.figWidth   = 15 / 4 * Fig.nTileX; 
    Fig.deltaX     = .4;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .6;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = 1;
    Fig.gapT       = 0; 
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [0.02 0];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [fig, ax] = tileAxes(Fig);
    set(fig, 'invertHardCopy', 'off')

    % Scatterplot of PC1
    set(gcf, 'currentAxes', ax(1, 1))
    scatter3(x(1, :), x(2, :), x(3, :), markerSize, zPC(:, 1), ...
              'filled' )
    plot3(x(1, idxTLim(1) : idxTLim(2)), ...
           x(2, idxTLim(1) : idxTLim(2)), ...
           x(3, idxTLim(1) : idxTLim(2)), 'k-', ...
           'lineWidth', 2)
              
    title('(a) PC_1 on L63 attracror')
    axis off
    view(0, 0)
    set(gca, 'cLim', [-1 1] * aPC(1)) 

    axPos = get(gca, 'position');
    hC = colorbar('location', 'southOutside');
    cPos = get(hC, 'position');
    cPos(2) = cPos(2) - .07;
    cPos(4) = .5 * cPos(4);
    set(hC, 'position', cPos)
    set(gca, 'position', axPos)


    % (PC1, PC2) angle
    set(gcf, 'currentAxes', ax(2, 1))
    scatter3(x(1, :), x(2, :), x(3, :), markerSize, anglPC, ...
              'filled' )
    plot3(x(1, idxTLim(1) : idxTLim(2)), ...
           x(2, idxTLim(1) : idxTLim(2)), ...
           x(3, idxTLim(1) : idxTLim(2)), 'k-', ...
           'lineWidth', 2)
    title('(b) (PC_1, PC_2) angle on L63 attractor')
    axis off
    view(90, 0)
    set(gca, 'cLim', [-pi pi ])
    colormap(gca, hsv)

    axPos = get(gca, 'position');
    hC = colorbar('location', 'southOutside');
    cPos = get(hC, 'position');
    cPos(2) = cPos(2) - .07;
    cPos(4) = .5 * cPos(4);
    set(hC, 'position', cPos)
    set(gca, 'position', axPos)
    set(hC, 'xTick', [-1 : .5 : 1] * pi, 'xTickLabel', ...
         {'-\pi' '-\pi/2' '0' '\pi/2' '\pi'}) 


    % (PC1,PC2) projection
    set(gcf, 'currentAxes', ax(3, 1))
    plot(zPC(idxTLim(1) : idxTLim(2), 1), ...
          zPC(idxTLim(1) : idxTLim(2), 2), ...
          'b-', 'linewidth', 1.5) 
    xlim([-2 2])
    ylim([-2 2])
    grid on
    title('(c) (PC_1, PC_2) projection')
    xlabel('PC_1')
    ylabel('PC_2')
    
    % PC1 time series
    set(gcf, 'currentAxes', ax(4, 1))
    plot(t, zPC(idxTLim(1) : idxTLim(2), 1), 'b-', ...
          'linewidth', 1.5)
    grid on
    xlim([t(1) t(end)])
    ylim([-2 2])
    title('(d) PC_1 time series')
    xlabel('t')

    % Re(g) scatterplot
    set(gcf, 'currentAxes', ax(1, 2))
    scatter3(x(1, :), x(2, :), x(3, :), markerSize, ...
              real(z(:, idxZ(1))), 'filled' )
    title(sprintf('(e) Re(g_{%i}) on L63 attractor', idxZ(1) - 1))
    axis off
    view(0, 0)
    set(gca, 'cLim', [-1 1] * a(idxZ))

    axPos = get(gca, 'position');
    hC = colorbar('location', 'southOutside');
    cPos = get(hC, 'position');
    cPos(2) = cPos(2) - .07;
    cPos(4) = .5 * cPos(4);
    set(hC, 'position', cPos)
    set(gca, 'position', axPos)

    % g phase angle
    set(gcf, 'currentAxes', ax(2, 2))
    scatter3(x(1, :), x(2, :), x(3, :), markerSize, angl, 'filled' )
    title(sprintf(['(f) g_{%i} phase angle L63 attractor'], ...
        idxZ(1) - 1))
    axis off
    view(90, 0)
    set(gca, 'cLim', [-pi pi ])
    colormap(gca, hsv)

    axPos = get(gca, 'position');
    hC = colorbar('location', 'southOutside');
    cPos = get(hC, 'position');
    cPos(2) = cPos(2) - .07;
    cPos(4) = .5 * cPos(4);
    set(hC, 'position', cPos)
    set(gca, 'position', axPos)
    set(hC, 'xTick', [-1 : .5 : 1] * pi, 'xTickLabel', ...
         {'-\pi' '-\pi/2' '0' '\pi/2' '\pi'}) 


    % g1 projection
    set(gcf, 'currentAxes', ax(3, 2))
    plot(z(idxTLim(1) : idxTLim(2), idxZ(1) ), ...
          'b-', 'linewidth', 1.5) 
    xlim([-1.5 1.5])
    ylim([-1.5 1.5])
    grid on
    title(sprintf('(g) g_{%i} phase space', idxZ(1) - 1))
    xlabel(sprintf('Re(g_{%i})', idxZ(1) - 1))
    ylabel(sprintf('Im(g_{%i})', idxZ(1) - 1))

    % z1 time series
    set(gcf, 'currentAxes', ax(4, 2))
    plot(t, real(z(idxTLim(1) : idxTLim(2), idxZ(1))), 'b-', ...
          'linewidth', 1.5)
    grid on
    xlim([t(1) t(end)])
    ylim([-1.5 1.5])
    title(sprintf('(h) Re(g_{%i}) time series', idxZ(1) - 1))
    xlabel('t')

    % Print figure
    if ifPrintFig
        figFile = fullfile(figDir, 'figEig.png');
        print(fig, figFile, '-dpng', '-r300') 
    end
end


%% MAKE EIGENFUNCTION MOVIE
if ifMovieEig
    
    % Retrieve source data and NLSA eigenfunctions
    x = getData(model.srcComponent);
    x = x(:, 1 + nShiftTakens : nSE + nShiftTakens);
    z = getKoopmanEigenfunctions(model);
    omega = getKoopmanEigenfrequencies(model);
    T = getKoopmanEigenperiods(model);

    % Construct coherent observables based on phi's
    a = max(abs(z), [], 1); % z amplitude
    angl = angle(z(:, idxZ(1))); % z phase

    % Construct coherent observables based on PCs
    zPC = PCA.v(1 + nShiftTakens : nSE + nShiftTakens, [1 2]) ...
        .* signPC / sqrt(2);
    anglPC = angle(complex(zPC(:, 1), zPC(:, 2))); % z phase
    aPC = max(abs(zPC), [], 1); % PC amplitude 

    % Determine number of movie frames; assign timestamps
    nFrame = idxTLim(2) - idxTLim(1) + 1;
    t = (0 : nFrame - 1) * In.dt;  

    % Set up figure and axes 
    Fig.nTileX     = 4;
    Fig.nTileY     = 2;
    Fig.units      = 'pixels';
    Fig.figWidth   = 1000; 
    Fig.deltaX     = 20;
    Fig.deltaX2    = 20;
    Fig.deltaY     = 50;
    Fig.deltaY2    = 30;
    Fig.gapX       = 50;
    Fig.gapY       = 70;
    Fig.gapT       = 40; 
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [0.02 0];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [fig, ax, axTitle] = tileAxes(Fig);

    % Set up videowriter
    movieFile = 'movieEig.mp4';
    movieFile = fullfile(figDir, movieFile);
    writerObj = VideoWriter(movieFile, 'MPEG-4');
    writerObj.FrameRate = 20;
    writerObj.Quality = 100;
    open(writerObj);


    % Loop over the frames
    for iFrame = 1 : nFrame

        iT = idxTLim(1) + iFrame - 1;

        % Scatterplot of PC1
        set(gcf, 'currentAxes', ax(1, 1))
        scatter3(x(1, :), x(2, :), x(3, :), markerSize, zPC(:, 1), ...
                  'filled' )
        scatter3(x(1, iT), x(2, iT), x(3, iT), 70, 'r', 'filled') 
        plot3(x(1, idxTLim(1) : iT), x(2, idxTLim(1) : iT), ...
               x(3, idxTLim(1) : iT), 'k-', ...
                'lineWidth', 1.5)
                  
        title('(a) PC1 on L63 attracror')
        axis off
        view(0, 0)
        set(gca, 'cLim', [-1 1] * aPC(1)) 

        axPos = get(gca, 'position');
        hC = colorbar('location', 'southOutside');
        cPos = get(hC, 'position');
        cPos(2) = cPos(2) - .07;
        cPos(4) = .5 * cPos(4);
        set(hC, 'position', cPos)
        set(gca, 'position', axPos)


        % (PC1, PC2) angle
        set(gcf, 'currentAxes', ax(2, 1))
        scatter3(x(1, :), x(2, :), x(3, :), markerSize, anglPC, ...
                  'filled' )
        scatter3(x(1, iT), x(2, iT), x(3, iT), 70, 'k', 'filled') 
        plot3(x(1, idxTLim(1) : iT), x(2, idxTLim(1) : iT), ...
               x(3, idxTLim(1) : iT), 'k-', ...
                'lineWidth', 3)
        title('(b) (PC_1, PC_2) angle on L63 attractor')
        axis off
        view(90, 0)
        set(gca, 'cLim', [-pi pi ])
        colormap(gca, hsv)

        axPos = get(gca, 'position');
        hC = colorbar('location', 'southOutside');
        cPos = get(hC, 'position');
        cPos(2) = cPos(2) - .07;
        cPos(4) = .5 * cPos(4);
        set(hC, 'position', cPos)
        set(gca, 'position', axPos)
        set(hC, 'xTick', [-1 : .5 : 1] * pi, 'xTickLabel', ...
            {'-\pi' '-\pi/2' '0' '\pi/2' '\pi'}) 


        % (PC1,PC2) projection
        set(gcf, 'currentAxes', ax(3, 1))
        plot(zPC(idxTLim(1) : iT, 1), zPC(idxTLim(1) : iT, 2), ...
              'b-', 'linewidth', 1.5) 
        scatter(zPC(iT, 1), zPC(iT, 2), 70, 'r', 'filled') 
        xlim([-2 2])
        ylim([-2 2])
        grid on
        title('(c) (PC_1, PC_2) projection')
        xlabel('PC_1')
        ylabel('PC_2')

        % PC1 time series
        set(gcf, 'currentAxes', ax(4, 1))
        plot(t(1 : iFrame), zPC(idxTLim(1) : iT, 1), 'b-', ...
              'linewidth', 1.5)
        scatter(t(iFrame), zPC(iT, 1), 70, 'r', 'filled') 
        grid on
        xlim([t(1) t(end)])
        ylim([-2 2])
        title('(d) PC_1 time series')
        xlabel('t')

        % g scatterplot
        set(gcf, 'currentAxes', ax(1, 2))
        scatter3(x(1, :), x(2, :), x(3, :), markerSize, ...
                  real(z(:, idxZ(1))), 'filled' )
        scatter3(x(1, iT), x(2, iT), x(3, iT), 70, 'r', 'filled') 
        plot3(x(1, idxTLim(1) : iT), x(2, idxTLim(1) : iT), ...
               x(3, idxTLim(1) : iT), 'k-', ...
                'lineWidth', 2)
        title(sprintf('(e) Re(g_{%i}) on L63 attractor', idxZ(1) - 1))
        axis off
        view(0, 0)
        set(gca, 'cLim', [-1 1] * a(idxZ(1) ))

        axPos = get(gca, 'position');
        hC = colorbar('location', 'southOutside');
        cPos = get(hC, 'position');
        cPos(2) = cPos(2) - .07;
        cPos(4) = .5 * cPos(4);
        set(hC, 'position', cPos)
        set(gca, 'position', axPos)

        % g phase angle
        set(gcf, 'currentAxes', ax(2, 2))
        scatter3(x(1, :), x(2, :), x(3, :), markerSize, angl, 'filled' )
        scatter3(x(1, iT), x(2, iT), x(3, iT), 70, 'k', 'filled') 
        plot3(x(1, idxTLim(1) : iT), x(2, idxTLim(1) : iT), ...
               x(3, idxTLim(1) : iT), 'k-', ...
                'lineWidth', 3)
        title(sprintf(['(f) g_{%i} phase angle on L63 attractor'], ...
               idxZ(1) - 1))
        axis off
        view(90, 0)
        set(gca, 'cLim', [-pi pi ])
        colormap(gca, hsv)

        axPos = get(gca, 'position');
        hC = colorbar('location', 'southOutside');
        cPos = get(hC, 'position');
        cPos(2) = cPos(2) - .07;
        cPos(4) = .5 * cPos(4);
        set(hC, 'position', cPos)
        set(gca, 'position', axPos)
        set(hC, 'xTick', [-1 : .5 : 1] * pi, 'xTickLabel', ...
            {'-\pi' '-\pi/2' '0' '\pi/2' '\pi'}) 

        % g projection
        set(gcf, 'currentAxes', ax(3, 2))
        plot(real(z(idxTLim(1) : iT, idxZ(1))), ...
              imag(z(idxTLim(1) : iT, idxZ(1))), ...
              'b-', 'linewidth', 1.5) 
        scatter(real(z(iT, idxZ(1))), ...
                 imag(z(iT, idxZ(1))), 50, 'r', 'filled') 
        xlim([-1.5 1.5])
        ylim([-1.5 1.5])
        grid on
        title(sprintf('(g) g_{%i} phase space', idxZ(1) - 1))
        xlabel(sprintf('Re(g_{%i})', idxZ(1) - 1))
        ylabel(sprintf('Im(g_{%i})', idxZ(1) - 1))

        % Re(z1) time series
        set(gcf, 'currentAxes', ax(4, 2))
        plot(t(1 : iFrame), real(z(idxTLim(1) : iT, idxZ(1))), ...
              'b-', 'linewidth', 1.5)
        scatter(t(iFrame), real(z(iT, idxZ(1))), 50, 'r', 'filled') 
        grid on
        xlim([t(1) t(end)])
        ylim([-1.5 1.5])
        title(sprintf('(h) Re(g_{%i}) time series', idxZ(1) - 1))
        xlabel('t')

        title(axTitle, sprintf('t = %1.2f', t(iFrame)))
        axis(axTitle, 'off')
        
        frame = getframe(fig);
        writeVideo(writerObj, frame)
        
        for iY = 1 : Fig.nTileY
            for iX = 1 : Fig.nTileX
                cla(ax(iX, iY), 'reset')
            end
        end
        cla(axTitle, 'reset')
    end

    % Close video file and figure
    close(writerObj)
    close(fig)
end

        

%% AUXILIARY FUNCTIONS
%
% Function to compute phase composites from target data of NLSA model
function comp = computePhaseComposites(model, selectInd, iStart, iEnd, ...
                                        weights)
nC = size(model.trgComponent, 1); % number of observables to be composited
nPhase = numel(selectInd); % number of phases       
ifWeights = nargin == 5; % 

comp = cell(1, nC);

% Loop over the components
for iC = 1 : nC

    % Read data from NLSA model  
    y = getData(model.trgComponent(iC));
    y = y(:, iStart : iEnd); 
        
    nD = size(y, 1); % data dimension
    comp{iC} = zeros(nD, nPhase);

        % Loop over the phases
        for iPhase = 1 : nPhase

            % Compute phase conditional average
            if ifWeights
                comp{iC}(:, iPhase) = y * weights{iPhase};
            else    
                comp{iC}(:, iPhase) = ...
                    mean(y(:, selectInd{iPhase}), 2);
            end

        end
    end
end


