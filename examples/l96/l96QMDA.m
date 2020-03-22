%% Script to perform QM data assimilation for partially observed L96 
% A new Koopman operator is estimated for each forecast interval
% Measurements at multiple stpatial points are performed


%% MODEL AND ASSIMILATION PARAMETERS

idxH  = [ 1 21 ];   % observed state vector components
idxY  = [ 1 11 21 31 ]; % estimated state vector components 

% Full obs in training phase
experiment = '128k';    % NLSA basis functions
nL    = 128;% number of eigenfunctions
nQ    = 16; % quantization levels
nDT   = 100; % timesteps between obs
nDA   = 20; % number of data assimilation steps

% Partial obs with Takens in training phase 
%experiment = '128k';    % NLSA basis functions
%nL    = 801;% number of eigenfunctions
%nQ    = 32; % quantization levels
%nDT   = 100; % timesteps between obs
%nDA   = 20; % number of data assimilation steps



% Probability contour plot
%tLimPlt = [ 0 10 ];
%lbl    = '(a)';


% Probability contour plot
%tLimPlt = [ 10 20 ];
%lbl    = '(b)';

% Probability section plot
%tLimPlt = [ 0 20 ];
%iHPlt = 18; 
%lbl = '(c)';

% Entropy plot
tLimPlt = [ 0 20 ];
lbl = '(d)'; 


%% SCRIPT EXCECUTION OPTIONS
ifRead        = false; % read NLSA eigenfunctions 
ifProjObs     = false; % compute projection operators for observed components
ifProjEst     = false; % compute projection operators for estimated components
ifKoop        = false; % compute Koopman operators
ifDA          = false; % perform data assimilation
ifEnt         = false; % compute relative entropy skill scores
ifPlotContour = true;
ifPlotSection = false;  
ifPlotEnt     = false;  


%% BASIC ARRAYS/SIZES
b = linspace( 0, 1, nQ + 1 ); % CDF bins
nH = numel( idxH );
nY = numel( idxY );

%% BUILD NLSA MODEL
[ model, Pars ] = l96NLSAModel_den_ose( experiment );

%% READ DATA FROM MODEL
if ifRead
    tic
    disp( 'Eigenfunction data')
    [ phi, mu, lambda ] = getDiffusionEigenfunctions( model ); 
    x = getData( model.srcComponent );
    x = x( :, 1 + Pars.nXB : end - Pars.nXA - Pars.nEL );
    xH = x( idxH, : ); % observed components in training phase
    xY = x( idxY, : ); % estimated components in training phase 
    xO = getData( model.outComponent );
    xO = xO( :, 1 + Pars.nXB : end - Pars.nXA - Pars.nEL );
    nS = numel( mu );
    toc
end


%% PROJECTION OPERATORS
if ifProjObs
    tic
    disp( 'Projection operators for observed components' )
    aH = zeros( nH, nQ + 1 );
    EH = cell( nH, nQ );
    for iH = 1 : nH
        aH( iH, : ) =  ksdensity( xH( iH, : ), b, 'Function', 'icdf' );
        for iQ = 1 : nQ 
            if iQ == 1
                idxT = find( xH( iH, : ) < aH( iH, 2 ) ); % -inf bin
            elseif iQ == nQ 
                idxT = find( xH( iH, : ) >= aH( iH, end - 1 ) ); % +inf bin
            else
                idxT = find( xH( iH, : ) >= aH( iH, iQ ) ...
                           & xH( iH, : ) < aH( iH, iQ + 1 ) );
            end
            EH{ iH, iQ } = phi( idxT, 1 : nL )' * phi( idxT, 1 : nL ) / nS;
        end
    end
    toc
end
if ifProjEst
    tic
    disp( 'Projection operators for estimated components' )
    aY = zeros( nY, nQ + 1 );
    EY = cell( nY, nQ );
    for iY = 1 : nY
        aY( iY, : ) =  ksdensity( xY( iY, : ), b, 'Function', 'icdf' );
        for iQ = 1 : nQ 
            if iQ == 1
                idxT = find( xY( iY, : ) < aY( iY, 2 ) ); % -inf bin
            elseif iQ == nQ 
                idxT = find( xY( iY, : ) >= aY( iY, end - 1 ) ); % +inf bin
            else
                idxT = find( xY( iY, : ) >= aY( iY, iQ ) ...
                           & xY( iY, : ) < aY( iY, iQ + 1 ) );
            end
            EY{ iY, iQ } = phi( idxT, 1 : nL )' * phi( idxT, 1 : nL ) / nS;
        end
    end
    toc
end


%% KOOPMAN OPERATORS
if ifKoop
    tic
    disp( 'Koopman operators' )
    U = cell( 1, nDT );
    for iDT = 1 : nDT
        U{ iDT } = ...
        phi( 1 : end - iDT, 1 : nL )' * phi( iDT + 1 : end, 1 : nL ) / nS;
    end 
    toc
end

    
%% DATA ASSIMILATION
if ifDA 

    tic 
    disp( 'DA initialization' )
    rho0 = zeros( nL ); % initial state
    rho0( 1, 1 ) = 1;

    nT = nDA * nDT + 1;
    t = Pars.dt * ( 0 : nT - 1 );
    h = xO( idxH, 1 : nT ); % observed signal
    y = xO( idxY, 1 : nT ); % true signal

    p  = zeros( nQ, nY, nT + nDA ); % probability for estimated components to lie in a given bin
    tP = zeros( 1, nT + nDA  ); % time vector, including assimilation steps
    affilH = zeros( nH, nT + nDA ); % affiliation function for obs, including assimilation steps
    affilY = zeros( nY, nT + nDA ); % affiliation function for estimated components, including assimilation steps

    iT = 1;
    for iY = 1 : nY
        for iQ = 1 : nQ
            p( iQ, iY, 1 ) = trace( EY{ iY, iQ } * rho0 );
        end
        affilY( iY, 1 ) = discretize( y( iY, iT ), aY( iY, : ) );
    end
    for iH = 1 : nH
        affilH( iH, 1 ) = discretize( h( iH, iT ), aH( iH, : ) );
    end
    iP = 2;
    toc

    for iDA = 1 : nDA

        tic
        disp( [ 'DA step ' int2str( iDA ) '/' int2str( nDA ) ] )

        % unitary dynamics
        for iDT = 1 : nDT
           iT = iT + 1;
           rho = U{ iDT }' * rho0 * U{ iDT };
           rho = rho / trace( rho );
           for iY = 1 : nY
               for iQ = 1 : nQ
                   p( iQ, iY, iP ) = trace( EY{ iY, iQ } * rho ); 
                end
                affilY( iY, iP ) = discretize( y( iY, iT ), aY( iY, : ) );
            end
            tP( iP ) = tP( iP - 1 ) + Pars.dt;
            for iH = 1 : nH
                affilH( iH, iP ) = discretize( h( iH, iT ), aH( iH, : ) );
            end
            iP = iP + 1;
        end

        % projective dynamics
        for iH = 1 : nH
            iQ = discretize( h( iH, iT ), aH( iH, : ) );
            affilH( iH, iP ) = iQ;
            rho0 = EH{ iH, iQ } * rho * EH{ iH, iQ };
        end
        rho0 = rho0 / trace( rho0 );
        for iY = 1 : nY
            for iQ = 1 : nQ 
               p( iQ, iY, iP ) = trace( EY{ iY, iQ } * rho0 );
            end
            affilY( iY, iP ) = discretize( y( iY, iT ), aY( iY, : ) );
        end
        tP( iP ) = tP( iP - 1 );
        iP = iP + 1;
        toc
    end
end

if ifEnt
    dKL = zeros( nY, nT + nDA ); % precision
    eKL = zeros( nY, nT + nDA ); % accuracy
    for iP = 1 : nT + nDA
        for iY = 1 ; nY
            iSum = find( p( :, iY, iP ) > 0 );
            dKL( iY, iP ) = sum( p( iSum, iY, iP ) ...
                              .* log2( p( iSum, iY, iP ) * nQ ) ); 
            eKL( iY, iP ) = -log2( p( affilY( Y, iP ), iY, iP ) ); 
        end
    end
end


if ifPlotContour

    Mov.figWidth   = 6;    % in inches
    Mov.deltaX     = .5;
    Mov.deltaX2    = .7;
    Mov.deltaY     = .35;
    Mov.deltaY2    = .15;
    Mov.gapX       = .1;
    Mov.gapY       = .0;
    Mov.nSkip      = 5;
    Mov.cLimScl    = 1;

    nTileX = 1;
    nTileY = 1;

    panelX = ( Mov.figWidth - Mov.deltaX - Mov.deltaX2 - ( nTileX -1 ) * Mov.gapX ) / nTileX;
    panelY = panelX * ( 3 / 4 )^3;

    posn     = [ 0 ...
                 0 ...
                 nTileX * panelX + ( nTileX - 1 ) * Mov.gapX + Mov.deltaX + Mov.deltaX2 ...
                 nTileY * panelY + ( nTileY - 1 ) * Mov.gapY + Mov.deltaY + Mov.deltaY2 ];


    for iY = 1 : nY

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
                      'defaultAxesTickLength', [ 0.01 0 ], ...
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

        axPos = get( ax, 'position' );

        aGrd = aY( iY, : );
        aGrd( 1 ) = min( y( iY, : ) );
        aGrd( end ) = max( y( iY, : ) );

        [ tGrd, aGrd ] = meshgrid(  tP, ( aGrd( 1 : end - 1 ) + aGrd( 2 : end ) ) / 2 ); 
        pPlt = real( squeeze( p( :, iY, : ) ) ); 
        contourf( ax, tGrd, aGrd, log10( pPlt ), 4 );
        set( ax, 'tickDir', 'out', 'tickLength', [ 0.005 0 ] )
        set( ax, 'yLim', [ aY( iY, 1 ) aY( iY, end ) ], 'xLim', tLimPlt )
        %set( hP, 'edgeColor', 'none' )
        hC = colorbar( 'location', 'eastOutside' );
        cPos = get( hC, 'position' );
        cPos( 3 ) = .5 * cPos( 3 );
        cPos( 1 ) = cPos( 1 ) + .11;
        set( hC, 'position', cPos );
        ylabel( hC, 'log_{10}P' )
        set( ax, 'position', axPos );
        %set( gca, 'cLim', [ -7 0 ] )
        hold on
        plot( t, y( iY, : ), 'r-' )
        plot( t( nDT + 1 : nDT : end ), y( iY, nDT + 1  : nDT : end  ), 'r*' )
        xlabel( 't' )
        ylabel( 'y' )
        %text( -.08, 1.05, lbl, 'units', 'normalized' )
        %text( tPos( 1 ), tPos( 2 ), lbl )
        title( sprintf( 'Gridpoint %i/%i', idxY( iY ), Pars.n ) )
        %title( [ lblStr sprintf( ' \\Delta t / T = %1.2g', nDT * dt / 2 / pi * omega ) ] )

        print -dpng -r300 figPL63.png
    end
    
end



if ifPlotSection


    Mov.figWidth   = 6;    % in inches
    Mov.deltaX     = .5;
    Mov.deltaX2    = .7;
    Mov.deltaY     = .35;
    Mov.deltaY2    = .30;
    Mov.gapX       = .1;
    Mov.gapY       = .0;
    Mov.nSkip      = 5;
    Mov.cLimScl    = 1;

    nTileX = 1;
    nTileY = 1;

    panelX = ( Mov.figWidth - Mov.deltaX - Mov.deltaX2 - ( nTileX -1 ) * Mov.gapX ) / nTileX;
    panelY = panelX * ( 3 / 4 )^3;

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
                  'defaultAxesTickLength', [ 0.01 0 ], ...
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

    axPos = get( ax, 'position' );
    ifH = hO >= a( iHPlt ) ...
        & hO < a( iHPlt + 1 );
    idxH1 = find( ifH( 2 : end ) & ~ifH( 1 : end -1 ) );
    idxH2 = find( ifH( 1 : end - 1 ) & ~ifH( 2 : end ) );

    for iH = 1 : numel( idxH1 )
        hh = patch( [ t( idxH1( iH ) + 1 ) t( idxH1( iH ) + 1 ) t( idxH2( iH ) ) t( idxH2( iH ) ) ], ... 
                [ 0 1 1 0 ], 'green'  );
        set( hh, 'edgecolor', 'green' )
        hold on
    end

    tObs = t( nDT + 1 : nDT : end -1 );
    yVals = [ -0.2 : 0.1 : 1.2 ];
    tt = ones( size( yVals ) );
    for iObs = 1 : numel( tObs )
        plot( tObs( iObs ) * tt, yVals, 'r-' )
    end

    plot( tP, p( iHPlt, : ), 'b-' );
    set( ax, 'tickDir', 'out', 'tickLength', [ 0.005 0 ] )
    set( ax, 'yLim', [ 0 1 ], 'xLim', tLimPlt )
    %set( hP, 'edgeColor', 'none' )
    xlabel( 't' )
    ylabel( 'P_i' )
    text( -.08, 1.07, lbl, 'units', 'normalized' )
    title( sprintf( ' \\Xi_i  = [ %1.2f, %1.2f )', a( iHPlt ), a( iHPlt + 1 ) ) )


    print -dpng -r300 figL63_XSect.png
    
end


if ifPlotEnt


    Mov.figWidth   = 6;    % in inches
    Mov.deltaX     = .5;
    Mov.deltaX2    = .7;
    Mov.deltaY     = .35;
    Mov.deltaY2    = .20;
    Mov.gapX       = .1;
    Mov.gapY       = .0;
    Mov.nSkip      = 5;
    Mov.cLimScl    = 1;

    nTileX = 1;
    nTileY = 1;

    panelX = ( Mov.figWidth - Mov.deltaX - Mov.deltaX2 - ( nTileX -1 ) * Mov.gapX ) / nTileX;
    panelY = panelX * ( 3 / 4 )^3;

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
                  'defaultAxesTickLength', [ 0.01 0 ], ...
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

    plot( tP, eKL, 'g-' )
    hold on
    plot( tP, dKL, 'b-' )
    %legend( 'E', 'D' ) 
    %legend boxoff
    plot( tP, ones( 1, nT + nDA ) * log2( nQ ), 'm--' )
    tObs = t( nDT + 1 : nDT : end -1 );
    yVals = [ -0.2 : 1 : 20 ];
    tt = ones( size( yVals ) );
    for iObs = 1 : numel( tObs )
        plot( tObs( iObs ) * tt, yVals, 'r:' )
    end

    set( ax, 'tickDir', 'out', 'tickLength', [ 0.005 0 ] )
    set( ax, 'yLim', [ 0 10 ], 'xLim', tLimPlt )
    %set( hP, 'edgeColor', 'none' )
    xlabel( 't' )
    legend( 'E', 'D' )
    text( -.08, 1.07, lbl, 'units', 'normalized' )

    print -dpng -r300 figEntL63.png
    
end
