% DEMO OF KOOPMAN SPECTRAL ANALYSIS APPLIED TO VARIABLE-SPEED FLOW ON THE CIRCLE
%
% In this example, we consider the dynamical flow on the circle generated
% by the ODE
%
% dtheta/dt = f * (1 + sqrt(1 - a) * sin(theta))
% 
% theta(0) = 0,
%
% where theta is the phase angle on the circle,  f is a frequency parameter, 
% and a is a parameter in the interval (0, 1] controlling the nonlinearity of
% the flow.  
% 
% a = 1:     Constant-speed flow
% a = 0:     Singular flow with a fixed point at theta = 3 * pi / 2
% 0 < a < 1: Flow is faster for 0 < theta < pi and slower for pi < theta < 2 pi.
%
% The goal of this example is to demonstrate that the leading eigenfunction 
% of the Koopman generator of this system (acting on the L2 space associated
% with the invariant measure) rectifies the variable-speed flow, mapping
% the system to a conjugate system where the dynamics has constant frequency.
% The frequency of the rectified system is determined from the corresponding
% generator eigenvalue. 
%
% This script solves the eigenvalue problem for the generator in a data-driven
% basis obtained by eigenfunctions of a kernel integral operator (kernel 
% matrix), with a small amount of diffusion added for regularization.
%
% The kernel matrix is based on a variable-bandwidth Gaussian kernel with
% symmetric (bistochastic) Markov normalization. See Reference [2] below for
% further details and pseudocode. 
%
% The script calls function demoKoopman_nlsaModel to create an object of
% nlsaModel class, which encodes all aspects and parameters of the calculation,
% such as location of source data, kernel parameters, Koopman operator 
% approximation parameters, etc. 
%
% Results from each stage of the calculation are written on disk. Below is a
% summary of basic commands to access the code output:
%
% lambda = getDiffusionEigenvalues(model);    -- NLSA (kernel) eigenvalues
% phi    = getDiffusionEigenfunctions(model); -- kernel eigenfunctions
% z      = getKoopmanEigenfunctions(model);    -- Koopman eigenfunctions
% gamma  = getKoopmanEigenvalues(model);       -- Koopman eigenvalues  
% T      = getKoopmanEigenperiods(model);      -- Koopman eigenperiods
%
% The script has options to plot Fig. 4 and Movie 2 in Ref. [1], which can be 
% activated by setting ifPlotEig and ifMovieEig below to true, respectively.
%
% Figure/movie caption:
%
% Panel (a) shows the state space of the oscillator (i.e., the unit circle S^1)
% colored by the "original" x coordinate, f_orig(theta) = cos theta. The 
% dynamics is chosen such that dtheta / dt is larger for 0 < theta < pi and 
% smaller for pi < theta < 2 pi, resulting in the time f_orig(theta(t)) in 
% Panel (c).  
%
% Panel (d) shows the real part of the leading generator eigenfunction g_rect, 
% where it is evident that the values undergo a slower (faster) progression 
% when d theta / dt is high (slow). This leads to the rectified time series 
% g_rect(theta(t)) in Panel (f). The latter, is a pure cosine wave 
% cos(2 pi t / T) with period T = 4 determined from the generator eigenvalue 
% corresponding to g. 
%
% Panels (b) and (d) show f_orig and Re g_rect on the rectified state space 
% obtained by a nonlinear mapping (homeomorphism) h of the circle, using the 
% real and imaginary parts of g as coordinates. 
%
% In Panel~(e), the evolution of the phase angle in the rectified state space 
% is that of a harmonic oscillator with constant angular frequency 2 \pi / T.
% See Methods for further details on the dynamical system employed in this 
% example.
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
% Modified 2021/09/14

%% EXPERIMENT SPECIFICATION AND SCRIPT EXECUTION OPTIONS
experiment = 'a0.7';  

ifSourceData = true;  % generate source data
ifNLSA       = true;  % run NLSA (kernel eigenfunctions)
ifKoopman    = true;  % compute Koopman eigenfunctions 
ifPlotEig    = true;  % show dynamical rectification by generator eigenfunction
ifMovieEig   = false;  % make eigenfunction movie
ifPrintFig   = true;   % print figures to file

%% BATCH PROCESSING
iProc = 1; % index of batch process for this script
nProc = 1; % number of batch processes

%% GLOBAL PARAMETERS
% nShiftPlt:   Temporal shift applied to eigenfunctions to illustrate action
%              of Koopman operator
% idxZPlt:     Eigenfunctions to plot
% idxTPlt:     Time interval to plot
% idxTMrk:     Timestamp to mark in rectification plots
% phaseZ:      Phase factor for Koopman eigenfunctions
% idxZRec:     Eigenfunction to show in rectification figure/movie
% figDir:      Output directory for plots
% markerSize:  For eigenfunction scatterplots

switch experiment

    % Experiment with period tuned to match the El Nino Southern Oscillation
    case 'a0.7'
        idxZPlt    = [2 4];     
        phaseZ     = [1]; 
        nShiftPlt  = [0 500 1000]; 
        idxTPlt    = [1 2001]; 
        idxTMrk    = 745;
        idxZRec    = 2;
        markerSize = 7;         
        tSkipPlt   = 2;

    otherwise
        'Invalid experiment.'
end

% Figure directory
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

%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES
% In is a data structure containing the NLSA parameters for the training data.
%
% nSE is the number of samples avaiable for data analysis after Takens delay
% embedding.
%
% nSB is the number of samples left out in the start of the time interval (for
% temporal finite differnences employed in the kerenl).
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


%% EIGENFUNCTION FIGURE
if ifPlotEig
    
    % Retrieve source data and generator eigenfunctions. Assign timestamps.
    x = getData(model.srcComponent);
    x = x(:, 1 + nShiftTakens : nSE + nShiftTakens);
    z = getKoopmanEigenfunctions(model);
    t = (0 : nSE - 1) * In.Res.dt;  
    tPlt = t(idxTPlt(1) : idxTPlt(2));
    tPlt = tPlt - tPlt(1); % set time origin to 1st plotted point

    [~, iMax] = max(x(1, :));
    phaseFact = z(iMax, idxZRec);
    phaseFact = phaseFact / abs(phaseFact);
    zPlt = z(:, idxZRec) / phaseFact;

    % Set up figure and axes 
    Fig.nTileX     = 3;
    Fig.nTileY     = 2;
    Fig.units      = 'inches';
    Fig.figWidth   = 12 / 4 * Fig.nTileX; 
    Fig.deltaX     = .7;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .55;
    Fig.deltaY2    = .3;
    Fig.gapX       = .45;
    Fig.gapY       = 0.9;
    Fig.gapT       = 0; 
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [0.02 0];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [fig, ax] = tileAxes(Fig);
    set(fig, 'invertHardCopy', 'off')

    % f_orig on original (unrectified) state space
    set(gcf, 'currentAxes', ax(1, 1))
    scatter(x(1, :), x(2, :), markerSize, x(1, :), 'filled' )
    scatter(x(1, idxTMrk), x(2, idxTMrk ), 70, 'g', 'filled') 
    colormap(redblue)
    set(gca, 'cLim', [-1 1], ...
              'color', [1 1 1] * .3, ...
              'yTickLabel', [])
    xlim([-1.2 1.2])
    ylim([-1.2 1.2])
    title('(a) Re(f_{orig}) on original space')
    xlabel('Re(f_{orig})')

    axPos = get(gca, 'position');
    hC = colorbar('location', 'westOutside');
    cPos = get(hC, 'position');
    cPos(1) = cPos(1) - .07;
    cPos(3) = .5 * cPos(3);
    set(hC, 'position', cPos)
    set(gca, 'position', axPos)

    % f_orig on rectified state space
    set(gcf, 'currentAxes', ax(1, 2))
    scatter(real(zPlt), imag(zPlt), markerSize, x(1, :), 'filled' )
    scatter(real(zPlt(idxTMrk)), imag(zPlt(idxTMrk)), ...
             70, 'g', 'filled' )
    colormap(redblue)
    set(gca, 'cLim', [-1 1], ...
              'color', [1 1 1] * .3, ...
              'yTickLabel', [])
    xlim([-1.2 1.2])
    ylim([-1.2 1.2])
    title('(d) Re(f_{orig}\circ h^{-1}) on rectified space')
    xlabel(sprintf('Re(g_{rect})'))
        
    axPos = get(gca, 'position');
    hC = colorbar('location', 'westOutside');
    cPos = get(hC, 'position');
    cPos(1) = cPos(1) - .07;
    cPos(3) = .5 * cPos(3);
    set(hC, 'position', cPos)
    set(gca, 'position', axPos)

    % Time series of f_orig 
    set(gcf, 'currentAxes', ax(3, 1))
    plot(tPlt, x(1, idxTPlt(1) : idxTPlt(end)), '-', 'linewidth', 1)
    scatter(tPlt(idxTMrk), x(1, idxTMrk), 70, 'g', 'filled')
    grid on
    xlim([tPlt(1) tPlt(end)])
    ylim([-1.2 1.2])
    set(gca, 'xTick', tPlt(1) : tSkipPlt : tPlt(end))
    title('(c) Re(f_{orig}) time series')
    xlabel('t') 

    % Generator eigenfunction g_rect on original (unrectified) state space
    set(gcf, 'currentAxes', ax(2, 1))
    scatter(x(1, :), x(2, :), markerSize, real(zPlt), 'filled' )
    scatter(x(1, idxTMrk), x(2, idxTMrk ), 70, 'g', 'filled') 
    colormap(redblue)
    set(gca, 'cLim', max(abs(zPlt)) * [-1 1], ...
              'color', [1 1 1] * .3, ...
              'yTickLabel', [])
    xlim([-1.2 1.2])
    ylim([-1.2 1.2])
    title(sprintf('(b) Re(g_{%i}) on original state space' ,idxZRec - 1))
    title('(b) Re(g_{rect}\circ h) on original space')
    xlabel('Re(f_{orig})')
 
    % Generator eigenfunction on rectified state space
    set(gcf, 'currentAxes', ax(2, 2))
    scatter(real(zPlt), imag(zPlt), markerSize, real(zPlt), ...
            'filled' )
    scatter(real(zPlt(idxTMrk)), imag(zPlt(idxTMrk)), ...
             70, 'g', 'filled' )
    colormap(redblue)
    set(gca, 'cLim', max(abs(zPlt)) * [-1 1], ...
              'color', [1 1 1] * .3, ...
              'yTickLabel', [])
    xlim([-1.2 1.2])
    ylim([-1.2 1.2])
    title(sprintf('(e) Re(g_{rect}) on rectified space' ,idxZRec - 1))
    xlabel(sprintf('Re(g_{rect})'))
    
    % Time series of generator eigenfunction
    set(gcf, 'currentAxes', ax(3, 2))
    plot(tPlt, real(zPlt(idxTPlt(1) : idxTPlt(end))), ...
          '-', 'linewidth', 1)
    scatter(tPlt(idxTMrk), real(zPlt(idxTMrk)), 70, 'g', 'filled')
    grid on
    xlim([tPlt(1) tPlt(end)])
    set(gca, 'xTick', tPlt(1) : tSkipPlt : tPlt(end))
    ylim([-1.2 1.2])
    xlabel('t') 
    title(sprintf('(f) Re(g_{rect}) time series', idxZRec - 1))

    % Print figure
    if ifPrintFig
        figFile = 'figEig.png';
        figFile = fullfile(figDir, figFile);
        print(fig, figFile, '-dpng', '-r600') 
    end
end

%% MAKE MOVIE ILLUSTRATING RECTIFICATION
if ifMovieEig
    
    % Retrieve source data and generator eigenfunctions. Assign timestamps.
    x = getData(model.srcComponent);
    x = x(:, 1 + nShiftTakens : nSE + nShiftTakens);
    z = getKoopmanEigenfunctions(model);
    t = (0 : nSE - 1) * In.Res.dt;  
    tPlt = t(idxTPlt(1) : idxTPlt(2));
    tPlt = tPlt - tPlt(1); % set time origin to 1st plotted point
    
    [~, iMax] = max(x(1, :));
    phaseFact = z(iMax, idxZRec);
    phaseFact = phaseFact / abs(phaseFact);
    zPlt = z(:, idxZRec) / phaseFact;

    % Set up figure and axes 
    Fig.nTileX     = 3;
    Fig.nTileY     = 2;
    Fig.units      = 'pixels';
    Fig.figWidth   = 550; 
    Fig.deltaX     = 40;
    Fig.deltaX2    = 15;
    Fig.deltaY     = 40;
    Fig.deltaY2    = 20;
    Fig.gapX       = 40;
    Fig.gapY       = 60;
    Fig.gapT       = 20; 
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 8;
    Fig.tickLength = [0.02 0];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [fig, ax, axTitle] = tileAxes(Fig);
    colormap(redblue)

    % Set up videowriter
    movieFile = 'movieEig.mp4';
    movieFile = fullfile(figDir, movieFile);
    writerObj = VideoWriter(movieFile, 'MPEG-4');
    writerObj.FrameRate = 60;
    writerObj.Quality = 100;
    open(writerObj);

    % Determine number of movie frames; assign timestamps
    nFrame = idxTPlt(2) - idxTPlt(1) + 1;
    t = (0 : nFrame - 1) * In.Res.dt;  

    % Loop over the frames
    for iFrame = 1 : nFrame

        iT = idxTPlt(1) + iFrame - 1;

        % x1 observable on original (unrectified) state space
        set(gcf, 'currentAxes', ax(1, 1))
        scatter(x(1, :), x(2, :), markerSize, x(1, :), 'filled' )
        scatter(x(1, iT), x(2, iT), 70, 'g', 'filled') 
        colormap(redblue)
        set(gca, 'cLim', [-1 1], ...
                  'color', [1 1 1] * .3, ...
                  'yTickLabel', [])
        xlim([-1.2 1.2])
        ylim([-1.2 1.2])
        title('(a) Re(f_{orig}) on original space')
        xlabel('Re(f_{orig})')

        axPos = get(gca, 'position');
        hC = colorbar('location', 'westOutside');
        cPos = get(hC, 'position');
        cPos(1) = cPos(1) - .07;
        cPos(3) = .5 * cPos(3);
        set(hC, 'position', cPos)
        set(gca, 'position', axPos)

        % x1 observable on rectified state space
        set(gcf, 'currentAxes', ax(1, 2))
        scatter(real(zPlt), imag(zPlt), markerSize, x(1, :), 'filled' )
        scatter(real(zPlt(iT)), imag(zPlt(iT)), 70, 'g', 'filled') 
        set(gca, 'cLim', [-1 1], ...
                  'color', [1 1 1] * .3, ...
                  'yTickLabel', [])
        xlim([-1.2 1.2])
        ylim([-1.2 1.2])
        title('(d) Re(f_{orig}\circ h^{-1}) on rectified space')
        xlabel(sprintf('Re(g_{rect})'))
        
        axPos = get(gca, 'position');
        hC = colorbar('location', 'westOutside');
        cPos = get(hC, 'position');
        cPos(1) = cPos(1) - .07;
        cPos(3) = .5 * cPos(3);
        set(hC, 'position', cPos)
        set(gca, 'position', axPos)

        % Time series of x1 observable
        set(gcf, 'currentAxes', ax(3, 1))
        plot(tPlt(1 : iFrame), x(1, idxTPlt(1) : iT), '-', ...
              'linewidth', 1)
        scatter(tPlt(iFrame), x(1, iT), 50, 'g', 'filled')
        grid on
        xlim([tPlt(1) tPlt(end)])
        ylim([-1.2 1.2])
        set(gca, 'xTick', tPlt(1) : tSkipPlt : tPlt(end))
        title('(c) Re(f_{orig}) time series')
        xlabel('t') 

        % Generator eigenfunction on original (unrectified) state space
        set(gcf, 'currentAxes', ax(2, 1))
        scatter(x(1, :), x(2, :), markerSize, real(zPlt), 'filled' )
        scatter(x(1, iT), x(2, iT), 70, 'g', 'filled') 
        colormap(redblue)
        set(gca, 'cLim', max(abs(zPlt)) * [-1 1], ...
                  'color', [1 1 1] * .3, ...
                  'yTickLabel', [])
        xlim([-1.2 1.2])
        ylim([-1.2 1.2])
        title('(b) Re(g_{rect}\circ h) on original space')
        xlabel('Re(f_{orig})')
     
        % Generator eigenfunction on rectified state space
        set(gcf, 'currentAxes', ax(2, 2))
        scatter(real(zPlt), imag(zPlt), markerSize, real(zPlt), ...
                'filled' )
        scatter(real(zPlt(iT)), imag(zPlt(iT)), 70, 'g', 'filled') 
        colormap(redblue)
        set(gca, 'cLim', max(abs(zPlt)) * [-1 1], ...
                  'color', [1 1 1] * .3, ...
                  'yTickLabel', [])
        xlim([-1.2 1.2])
        ylim([-1.2 1.2])
        title(sprintf('(e) Re(g_{rect}) on rectified space' ,idxZRec - 1))
        xlabel(sprintf('Re(g_{rect})'))
        
        % Time series of generator eigenfunction
        set(gcf, 'currentAxes', ax(3, 2))
        plot(tPlt(1 : iFrame), real(zPlt(idxTPlt(1) : iT)), '-', ...
              'linewidth', 1)
        scatter(tPlt(iFrame), real(zPlt(iT)), 50, 'g', 'filled')
        grid on
        xlim([tPlt(1) tPlt(end)])
        ylim([-1.2 1.2])
        set(gca, 'xTick', tPlt(1) : tSkipPlt : tPlt(end))
        xlabel('t') 
        title(sprintf('(f) Re(g_{%i}) time series', idxZRec - 1))

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
