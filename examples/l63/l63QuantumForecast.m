% SCRIPT TO COMPUTE APPROXIMATE KOOPMAN EIGENFUNCTIONS USING THE NLSA 
% BASIS
%
% Modified 2017/02/23

%% MAIN CALCULATION PARAMETERS AND OPTIONS
experiment = '6.4k'; 

% for forecasting
idxPhi     = 1 : 201;   % NLSA eigenfunctions 
idxZeta    = 1 : 101;   % generator eigenfunctions for quantum system
tau        = 1E-5;      % RKHS regularization parameter
tauRef     = 1E-4;      % diffusion regularization parameter
idxX       = [ 1 : 3 ]; % state vector components to predict
nP         = 500 + 1;   % prediction timesteps (including 0)
nPar       = 4;         % number of parallel workers
nProcX     = 1;         % number of batch processes for verification data
nProcT     = 250;       % number of processes for forecast times

idxT0 = 101; % forecast initialization time to plot

%% SCRIPT EXCECUTION OPTIONS
ifRead            = true;  % read data and eigenfunctions
ifCalcGenerator   = true;  % compute Koopman eigenvalues and eigenfunctions
ifCalcObservables = true;  % compute quantum observable operators
ifPred            = false;  % perform prediction
ifErr             = false;  % compute prediction errors 

%% BUILD NLSA MODEL, DETERMINE BASIC ARRAY SIZES
[ model, Pars ] = l63NLSAModel( experiment );
nX    = numel( idxX );    % dimension of prediction observable
nPhi  = numel( idxPhi );  % number of NLSA eigenfunctions
nZeta = numel( idxZeta ); % number of generator eigenfunctions
nR    = size( model.embComponent, 2 );    % training realizations
nS    = getNTotalSample( model.embComponent( 1, : ) );    % training samples
nSA   = sum( getNXA( model.trgEmbComponent( 1, : ) ) );   % extra samples
nT    = nS / nR; % temporal training samples
nD    = sum( getDataSpaceDimension( model.trgEmbComponent( :, 1 ) ) ); 
nRO   = size( model.outEmbComponent, 2 ); % verification realizations
nSO   = getNTotalSample( model.outEmbComponent( 1, : ) ); % test samples  
nBO   = getNTotalBatch( model.outEmbComponent( 1, : ) ); % test batches
nTO   = nSO / nRO; % temporal test samples

%% READ DATA FROM MODEL
if ifRead
    disp( 'Reading data...')
    tic
    % In-sample eigenfunctions
    [ phi, mu, lambda ] = getDiffusionEigenfunctions( model ); 

    % Out-of-samplel eigenfunctions
    phiO = getOseDiffusionEigenfunctions( model );

    % Training data, arranged into array of size [ nS nD ]
    f = getData( model.trgEmbComponent )';

    % Compute mean and standard deviation
    fMean = mean( f, 1 );   
    fStd  = std( f, 0, 1 );

    % Verification data, arranged into array of size [ nD, nSO + nSA ] 
    % Append data after main time interval to compute forecast error
    fOut = zeros( nD, nTO + nTA, nRO ); 
    fOut( :, 1 : nTO, : )  = reshape( ...
        getData( model.outTrgEmbComponent, [ nD nTO nRO ] ) );
    fOut( :, nTO + 1 : end, : ) = reshape( ...
        getData_after( model.outTrgEmbComponent, [ nD nTA nRO ] );
    fOut = reshape( fOut, [ nD nSO ] );
    toc
end

%% COMPUTE RKHS EIGENVALUES
% sqrtLambda: Square roots of RKHS eigenvalues, size [ nPhi 1 ]
epsilon = getBandwidth( model.diffOp );
eta = ( lambda( 1 ) ./ lambda - 1 ) / epsilon;
sqrtLambda = exp( - tau * eta( idxPhi ) / 2 );

%% EIGENVALUE PROBLEM FOR KOOPMAN GENERATOR 
%  Output arrays are as follows:
%  omega:  Koopman eigenfrequencies
%  zeta:   Koopman eigenfunctions
%  E:      Dirichlet energies of Koopman eigenfunctions
if ifCalcGenerator
    disp( 'Computing generator matrix...' )
    tic
    dphi  = 0.5 * ( reshape( phi( 3 : end, idxPhi ), [ nT - 2, nR, nPhi ] ) ...
              - reshape( phi( 1 : end - 2, idxPhi ), [ nT - 2, nR, nPhi ] ) ); 
    dPhi = reshape( dPhi, [ nS - 2, nPhi ] );
    phiMu = phi( 2 : end - 1, idxPhi ) .* mu( 2 : end - 1 );
    W = phiMu' * dphi / Pars.dt;
    W = sqrtLambda .* W .* sqrtLambda'; 
    W = .5 * ( W - W' );
    toc
    
    disp( 'Computing generator eigenfunctions...' )
    tic
    % Solve eigenvalue problem for generator matrix
    [ c, omega ] = eig( W );
    omega = imag( diag( omega ) ).'; % row vector
    
    % Compute Dirichlet energies and sort in increasing order 
    l2SqNorm = sum( abs( c .* sqrtLambda ) .^ 2, 1 );
    E = ( 1 ./ l2SqNorm - 1 ) ./ ( 1 - omega .^ 2 .* Pars.dt .^ 2 );
    [ E, idxE ] = sort( E, 'ascend' );
    c = c( :, idxE );
    l2SqNorm = l2SqNorm( idxE );
    toc
    
end

%% CALCULATE QUANTUM MECHANICAL OBSERVABLES
% Output: cell array Tf of size [ 1 nD ] containing operator matrices, 
% each of size [ nZeta nZeta ]
if ifCalcObservables
    disp( 'Computing quantum mechanical observables...' )
    tic
    Tf = cell( 1, nD ); % compactified multiplication operators
    cL = c( :, idxZeta ) .* sqrtLambda;   
    for iD = 1 : nD
        Tf{ iD } =  phi( :, idxPhi )' ...
                  * ( f( :, iD ) .* phi( :, idxPhi ) .* mu );
        Tf{ iD } = cL' * Tf{ iD } * cL;
    end
    toc
end

%% PERFORM PREDICTION
% We use up to rank-4 arrays for numerical efficiency (at the expense of 
% memory use). The size convention is [ nZeta nZeta nTO nP ]. 
%
% Forecast is output in an array fPred of size [ nTO nP nD ].
if ifPred
    
    % Partition the forecast interval into batches

    disp( 'Forming forecast operators...' )
    tic

    % Eigenfunction values at verification dataset 
    zetaO = phiO( :, idxPhi  ) .* sqrtLambda' * c;
    zetaO = zetaO.'; % size [ nZeta nSO ]

    % Forecast times
    t = ( 0 : nP - 1 ) * Pars.dt; 

    % Heisenberg operator
    Ut = omega - omega';
    Ut = exp( i * ( omega - omega' ) .* reshape( t, [ 1 1 1 nP ] ) );

    % K is a [ nZeta nZeta nSO ] array containing the summands (features) 
    % in the Mercer sum of the kernel at the verification points. 
    K = reshape( conj( zetaO ), [ nZeta 1 nSO ] ) ...
      .* reshape( zetaO, [ 1 nZeta nSO ] ); 

    % Product of Heisenberg operator and Mercer  
    KUt = Ut .* K;
    clear omega2 Ut
    
    % Kernel values
    K = sum( K, [ 1 2 ] );  

    % Predicted values
    fPred = zeros( nTO, nP, nD );
    toc

    % Evaluate prediction
    disp( 'Performing prediction...' )
    tic
    for iD = 1 : nD
       fPred( :, :, iD ) = sum( Tf{ iD } .* KUt, [ 1 2 ] ); 
    end
    fPred = fPred ./ K;
    toc
end

%% COMPUTE PREDICTION ERROR
if ifErr
    disp( 'Prediction error' )

    tic
    % create true signal by time-shifting out-of-sample data
    fTrue = lembed( fOut, [ nP, nP + nTO - 1 ], 1 : nP );
    
    % Put in appropriate form for comparison with fPred 
    fTrue = reshape( fTrue, [ nD nP nTO ] );
    fTrue = permute( fTrue, [ 3 2 1 ] );

    % Compute normalized RMSE 
    predErr = fPred - fTrue;
    fRmse = vecnorm( predErr, 2, 1 ) / sqrt( nSO );
    fRmse = fRmse ./ fStd; 

    toc
end

return

if ifPlotZetaJoint
    Mov.figWidth   = 8;    % in inches
    Mov.deltaX     = .45;
    Mov.deltaX2    = .15;
    Mov.deltaY     = .35;
    Mov.deltaY2    = .35;
    Mov.gapX       = .1;
    Mov.gapY       = .0;
    Mov.nSkip      = 5;
    Mov.cLimScl    = 1;


    x = getData( model.srcComponent );
    x = x( :, 1 + Pars.nXB : end - Pars.nXA );
    
    nTileX = numel( idxZetaPlt );
    nTileY = 2;

    panelX = ( Mov.figWidth - Mov.deltaX - Mov.deltaX2 - ( nTileX -1 ) * Mov.gapX ) / nTileX;
    panelY = panelX * 3 / 4;

    posn     = [ 0 ...
                 0 ...
                 nTileX * panelX + ( nTileX - 1 ) * Mov.gapX + Mov.deltaX + Mov.deltaX2 ...
                 nTileY * panelY + ( nTileY - 1 ) * Mov.gapY + Mov.deltaY + Mov.deltaY2 ];


    fig = figure( 'units', 'inches', ...
                  'paperunits', 'inches', ...
                  'position', posn, ...
                  'paperPosition', posn, ...
                  'color', 'white', ...
                  'doubleBuffer', 'on', ...
                  'backingStore', 'off', ...
                  'defaultAxesTickDir', 'out', ...
                  'defaultAxesNextPlot', 'replace', ...
                  'defaultAxesBox', 'on', ...
                  'defaultAxesFontSize', 8, ...
                  'defaultTextFontSize', 8, ...
                  'defaultAxesTickDir',  'out', ...
                  'defaultAxesTickLength', [ 0.02 0 ], ...
                  'defaultAxesFontName', 'helvetica', ...
                  'defaultTextFontName', 'helvetica', ...
                  'defaultAxesLayer', 'top' );

    ax = zeros( nTileX, nTileY );

    for iAx = 1 : nTileX
        for jAx = 1 : nTileY
            ax( iAx, jAx ) = axes( ...
                    'units', 'inches', ...
                    'position', ...
                    [ Mov.deltaX + ( iAx - 1 ) * ( panelX + Mov.gapX ), ...
                      Mov.deltaY + ( nTileY - jAx ) * ( panelY + Mov.gapY ), ...
                      panelX, panelY ] );
        end
    end

    for iZeta = 1 : nTileX
       tPlt = Pars.dt * ( 0 : numel( idxTPlt{ iZeta } ) - 1 );
       set( gcf, 'currentAxes', ax( iZeta, 1 ) )
       scatter3( x( 1, : ), x( 2, : ), x( 3, : ), 5,  real( zeta( :, idxZetaPlt( iZeta ) ) ), '.' )
       axis off
       view( 35, 15 )
       c0 = min( max( real( zeta( :, idxZetaPlt( iZeta ) ) ) ), ...
                max( -real( zeta( :, idxZetaPlt( iZeta ) ) ) ) );
       set( gca, 'cLim', [ -c0 c0 ] )
       title( sprintf( '\\zeta_{%i}, \\omega_{%i} = %1.3g, D(\\zeta_{%i}) = %#1.3g', ...
           idxZetaPlt( iZeta ), idxZetaPlt( iZeta ), ...
           omega( idxZetaPlt( iZeta ) ), ...
           idxZetaPlt( iZeta ), E( idxZetaPlt( iZeta ) ) ) )
       axPos = get( gca, 'position' );
       hC = colorbar( 'location', 'westOutside' );
       cPos = get( hC, 'position' );
       cPos( 3 ) = .5 * cPos( 3 );
       cPos( 1 ) = cPos( 1 ) - .04;
       cPos( 4 ) = .8 * cPos( 4 );
       cPos( 2 ) = cPos( 2 ) + .04;
       set( hC, 'position', cPos );
       set( gca, 'position', axPos );

       set( gcf, 'currentAxes', ax( iZeta, 2 ) )
       scl = max( real( zeta( idxTPlt{ iZeta }, idxZetaPlt( iZeta ) ) ) );
       plot( tPlt, real( zeta( idxTPlt{ iZeta }, idxZetaPlt( iZeta ) ) ) / scl )
       xlabel( 't' )
       if iZeta == 1
           ylabel( 'Re(\zeta)' )
       else
           set( gca, 'yTickLabel', [] )
       end
       set( gca, 'xLimSpec', 'tight' )
       set( gca, 'yLim', [ -1.2 1.2 ] )
   end
   print -dpng -r300 figZetaL63.png
       
end

if ifPlotPred
    fig = figure;
    tVals = ( idxTPltPred - 1 ) * Pars.dt;
    plot( tVals, fTrue( idxT0Plt, :, idxXPlt ), 'r-' )    
    hold on
    plot( tVals, fPred( idxT0Plt, :, idxXPlt ), 'b-' )
    legend( 'truth', 'prediction' )
    xlabel('t')
    ylabel('x_1')
    set( gca, 'xLim', [ tVals( 1 ) tVals( end ) ] )
    fName =  'figL63_pred.png';
    print( '-dpng', '-r150', fName )
end
    
if ifPlotPredErr
    fig = figure;
    tVals = ( idxTPltPred - 1 ) * Pars.dt;
    plot( tVals, predErrL2(  :, idxXPlt ), 'b-' )
    xlabel('t')
    ylabel('L2 error')
    set( gca, 'xLim', [ tVals( 1 ) tVals( end ) ] )
    fName = 'figL63_predErr.png';
    print( '-dpng', '-r150', fName )
end

if ifPlotPredJoint

    Mov.figWidth   = 8;    % in inches
    Mov.deltaX     = .45;
    Mov.deltaX2    = .15;
    Mov.deltaY     = .35;
    Mov.deltaY2    = .25;
    Mov.gapX       = .35;
    Mov.gapY       = .5;
    Mov.nSkip      = 5;
    Mov.cLimScl    = 1;


    nTileX = 3;
    nTileY = 2;

    panelX = ( Mov.figWidth - Mov.deltaX - Mov.deltaX2 - ( nTileX -1 ) * Mov.gapX ) / nTileX;
    panelY = panelX * 3 / 4;

    posn     = [ 0 ...
                 0 ...
                 nTileX * panelX + ( nTileX - 1 ) * Mov.gapX + Mov.deltaX + Mov.deltaX2 ...
                 nTileY * panelY + ( nTileY - 1 ) * Mov.gapY + Mov.deltaY + Mov.deltaY2 ];


    fig = figure( 'units', 'inches', ...
                  'paperunits', 'inches', ...
                  'position', posn, ...
                  'paperPosition', posn, ...
                  'color', 'white', ...
                  'doubleBuffer', 'on', ...
                  'backingStore', 'off', ...
                  'defaultAxesTickDir', 'out', ...
                  'defaultAxesNextPlot', 'replace', ...
                  'defaultAxesBox', 'on', ...
                  'defaultAxesFontSize', 8, ...
                  'defaultTextFontSize', 8, ...
                  'defaultAxesTickDir',  'out', ...
                  'defaultAxesTickLength', [ 0.02 0 ], ...
                  'defaultAxesFontName', 'helvetica', ...
                  'defaultTextFontName', 'helvetica', ...
                  'defaultAxesLayer', 'top' );

    ax = zeros( nTileX, nTileY );

    for iAx = 1 : nTileX
        for jAx = 1 : nTileY
            ax( iAx, jAx ) = axes( ...
                    'units', 'inches', ...
                    'position', ...
                    [ Mov.deltaX + ( iAx - 1 ) * ( panelX + Mov.gapX ), ...
                      Mov.deltaY + ( nTileY - jAx ) * ( panelY + Mov.gapY ), ...
                      panelX, panelY ] );
        end
    end

    for iX = 1 : 3

        tVals = ( idxTPltPred - 1 ) * Pars.dt;

        set( gcf, 'currentAxes', ax( iX, 1 ) )
        plot( tVals, fTrue( idxT0Plt, idxTPltPred, iX ), 'r-' )    
        hold on
        plot( tVals, fPred( idxT0Plt, idxTPltPred, iX ), 'b-' )
        if iX == 1
            legend( 'truth', 'prediction', 'location', 'northwest' )
            legend boxoff
        end
        xlabel( 't' )
        title( sprintf( 'F_%i', iX ) )
        set( gca, 'xLimSpec', 'tight' )

        set( gcf, 'currentAxes', ax( iX, 2 ) )
        plot( tVals, predErrL2(  idxTPltPred, iX ), 'b-' )
        set( gca, 'xLimSpec', 'tight' )
        set( gca, 'yLim', [ 0 1.4 ], 'yTick', [ 0 : .2 : 1.4 ] )
        grid on
        xlabel( 'lead time t' )
        if iX == 1
            ylabel( 'normalized L2 error' )
        end
    end

    print -dpng -r300 figL63Pred_rkhs.png
end




