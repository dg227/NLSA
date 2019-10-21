%% PERFORM QUANTUM MECHANICAL DATA ASSIMILATION OF NOAA DATA

%% MODEL AND ASSIMILATION PARAMETERS
% Observed and estimated components are as follows
% 1 -> Nino 3.4 index
% 2 -> Nino 3.4 region SST
% 3 -> Nino 1+2 region SST
% 4 -> Nino 3 region SST
% 5 -> Nino 4 region SST
experiment = 'ip_sst_control_nino'; % NLSA model
idxH  = [ 1 : 5 ];     % observed  components
idxY  = [ 1 ];         % estimated  components 
idxR  = 1;             % realization (ensemble member) in assimilation phase    
nL    = 3001;           % number of eigenfunctions
nQ    = 7;             % quantization levels
nDT   = 1;             % timesteps between obs
nDTF  = 12;            % number of forecast timesteps (must be at least nDTF)
nPar  = 4;             % number of workers 

%tLimPlt = {  '200801' '201906' }; % time limit to plot
%tLimPlt = {  '120001' '124912' }; % time limit to plot
idxTPlt = [ 0 : 3 : 12 ] + 1; % lead times for running forecast
idxYPlt  = 1; % estimated components for running probability forecast


%% SCRIPT EXCECUTION OPTIONS
ifRead        = true; % read NLSA eigenfunctions 
ifProjObs     = true; % compute projection operators for observed components
ifProjEst     = true; % compute projection operators for estimated components
ifKoop        = true; % compute Koopman operators
ifDA          = true; % perform data assimilation
ifEnt         = true; % compute relative entropy skill scores

ifPlotProb  = true; % probability contours 
ifPlotEnt   = true; % entropy plots 
ifMovieProb = false; % histogram movie


%% BUILD NLSA MODEL, SETUP BASIC ARRYAS
%[ model, Pars, ParsO ] = climateNLSAModel_den_ose( experiment );
[ model, Pars, ParsO ] = climateNLSAModel_ose( experiment );
nH   = numel( idxH );
nY   = numel( idxY );
nS   = getNTotalEmbSample( model ); % training samples
nSO  = getNTotalOutSample( model ); % verification samples 
idxTObs = 1 : nDT : nSO;
nDA  = numel( idxTObs ); % data assimilation cycles
idxT1 = Pars.Res.idxT1 - floor( max( Pars.nE, Pars.nET ) / 2 );
%idxT1 = Pars.Res.idxT1;
b    = linspace( 0, 1, nQ + 1 ); % CDF bins

%% READ DATA FROM MODEL
if ifRead
    tic
    disp( 'Data retrieval')
    % Eigenfunctions
    [ phi, mu, lambda ] = getDiffusionEigenfunctions( model ); % eignefunctions
    phiMu = bsxfun( @times, phi, mu );

    % Training signal
    x  = getData( model.trgComponent );
    x  = x( :, idxT1 : idxT1 + nS - 1 ); % training signal
    h = x( idxH, : ); % observed components in training phase
    y = x( idxY, : ); % estimated components in training phase 

    % Assimilation signal
    x = zeros( ParsO.nCT, nSO );
    tStr = [ ParsO.Res( idxR ).tLim{ 1 } '-' ParsO.Res( idxR ).tLim{ 2 } ];
    for iC = 1 : ParsO.nCT
        xyStr = sprintf( 'x%i-%i_y%i-%i', ParsO.Trg( iC ).xLim( 1 ), ...
                                          ParsO.Trg( iC ).xLim( 2 ), ...
                                          ParsO.Trg( iC ).yLim( 1 ), ...
                                          ParsO.Trg( iC ).yLim( 2 ) );

        pathOut = fullfile( pwd, ...
                            'data/raw',  ...
                            ParsO.Res( idxR ).experiment, ...
                            ParsO.Trg( iC ).field,  ...
                            [ xyStr '_' tStr ] );

            
        xTmp = load( fullfile( pathOut, 'dataX.mat' ), 'x' );
        x( iC, :) = xTmp.x;
    end
    hO = x( idxH, : ); % observed components in assimilation phase
    yO = x( idxY, : ); % estimated components in assimilation phase
    yObs = yO( :, idxTObs ); 
    tNumOut = getOutTime( model ); % serial time numbers for true signal
    tNumObs = datemnth( tNumOut( 1 ), idxTObs - 1 )'; % initialization time  
    tNumVer = repmat( tNumObs, [ 1 nDTF + 1 ] ); % verification time
    tNumVer = datemnth( tNumVer, repmat( 0 : nDTF, [ nDA 1 ] ) )';
    toc
end


%% PROJECTION OPERATORS
if ifProjObs
    tic
    disp( 'Projection operators for observed components' )
    aH = zeros( nH, nQ + 1 );
    EH = cell( nH, nQ );
    parfor( iH = 1 : nH, nPar )
        aHLoc =  ksdensity( h( iH, : ), b, 'Function', 'icdf' );
        for iQ = 1 : nQ 
            if iQ == 1
                idxT = find( h( iH, : ) < aHLoc( 2 ) ); % -inf bin
            elseif iQ == nQ 
                idxT = find( h( iH, : ) >= aHLoc( end - 1 ) ); % +inf bin
            else
                idxT = find( h( iH, : ) >= aHLoc( iQ ) ...
                           & h( iH, : ) < aHLoc( iQ + 1 ) );
            end
            EH{ iH, iQ } = phi( idxT, 1 : nL )' * phiMu( idxT, 1 : nL );
        end
        aH( iH, : ) =  aHLoc;
    end
    toc
end

if ifProjEst
    tic
    disp( 'Projection operators for estimated components' )
    aY = zeros( nY, nQ + 1 );
    EY = cell( nY, nQ );
    parfor( iY = 1 : nY, nPar )
    %for iY = 1 : nY
        aYLoc = ksdensity( y( iY, : ), b, 'Function', 'icdf' );
        aYSav = aYLoc;
        for iQ = 1 : nQ 
            if iQ == 1
                idxT = find( y( iY, : ) < aYLoc( 2 ) ); % -inf bin
            elseif iQ == nQ 
                idxT = find( y( iY, : ) >= aYLoc( end - 1 ) ); % +inf bin
            else
                idxT = find( y( iY, : ) >= aYLoc( iQ ) ...
                           & y( iY, : ) < aYLoc( iQ + 1 ) );
            end
            EY{ iY, iQ } = phi( idxT, 1 : nL )' * phiMu( idxT, 1 : nL );
        end
        aY( iY, : ) =  aYLoc;
    end
    toc
end


%% KOOPMAN OPERATORS
if ifKoop
    tic
    disp( 'Koopman operators' )
    U = cell( 1, nDTF );
    parfor( iDT = 1 : nDTF, nPar )
        U{ iDT } = ...
        phi( 1 : end - iDT, 1 : nL )' * phiMu( iDT + 1 : end, 1 : nL );
    end 
    toc
end

%% DATA ASSIMILATION
if ifDA 

    p  = zeros( nQ, nY, nDTF + 1, nDA ); % forecast pro. for estimated compononents
    affilH = zeros( nH, nDA );       % affiliation function for obs

    tic 
    disp( 'DA initialization' )
    rho0 = zeros( nL ); % initial density matrix
    rho0( 1, 1 ) = 1;
    rho1 = rho0; % initialize this array here for use within parfor loop

    % Compute affiliation function and measurement  probabilities based on rho0
    for iH = 1 : nH
        affilH( iH, 1 ) = discretize( hO( iH, idxTObs( 1 ) ), aH( iH, : ) );
    end
    for iY = 1 : nY
        for iQ = 1 : nQ
            p( iQ, iY, 1, 1 ) = trace( EY{ iY, iQ } * rho0 );
        end
    end
    toc

    tic
    disp( 'DA main loop' )
    for iDA = 1 : nDA
        %tic
        disp( [ 'Step ' int2str( iDA ) '/' int2str( nDA ) ] )

        
        % Evolve density matrix by Koopman operators; store result from last iteration as rho1
        % This operation can be performed in parallel, as we are not 
        % updating the initial state
        % We also compute the affiliation function for the estimated components
        parfor( iDT = 1 : nDTF, nPar )
        %for iDT = 1 : nDTF
           iT = idxTObs( iDA ) + iDT;
           rho = U{ iDT }' * rho0 * U{ iDT };
           rho = rho / trace( rho );
           if iDT == nDTF
               rho1 = rho;
           end
           for iY = 1 : nY
               for iQ = 1 : nQ
                    p( iQ, iY, iDT + 1, iDA ) = trace( EY{ iY, iQ } * rho );
                end
           end
        end
        
        % Update state and obs affiliation function 
        rho0 = rho1; 
        if iDA < nDA
            for iH = 1 : nH
                affilH( iH, iDA + 1 ) = discretize( hO( iH, idxTObs( iDA ) ), aH( iH, : ) );
                rho0 = EH{ iH, affilH( iH, iDA + 1 ) } * rho0 * EH{ iH, affilH( iH, iDA + 1 ) };
            end
            rho0 = rho0 / trace( rho0 );
            for iY = 1 : nY
                for iQ = 1 : nQ
                    p( iQ, iY, 1, iDA + 1 ) = trace( EY{ iY, iQ } * rho0 );
                end
            end
        end
        
        %toc
    end
    toc

end


if ifEnt
    tic
    disp( 'Entropy metrics' )
    affilY = zeros( nY, nDTF + 1, nDA ); % affil. function for estimated components
    dKL = zeros( nY, nDTF + 1, nDA ); % precision
    eKL = zeros( nY, nDTF + 1, nDA ); % accuracy
    for iDA = 1 : nDA
        for iDT = 1 : nDTF + 1
            iT = idxTObs( iDA ) + iDT;
            for iY = 1 : nY
                iSum = find( p( :, iY, iDT, iDA ) > 0 );
                dKL( iY, iDT, iDA ) = sum( ...
                    p( iSum, iY, iDT, iDA ) .* log2( p( iSum, iY, iDT, iDA ) * nQ ) ); 
            end
            if iT <= nSO
                for iY = 1 : nY 
                    affilY( iY, iDT, iDA ) = discretize( yO( iY, iT ), aY( iY, : ) );
                    eKL( iY, iDT, iDA ) = -log2( p( affilY( iY, iDT, iDA ), iY, iDT, iDA ) ); 
                end
            else
                affilY( :, iDT, iDA ) = NaN;
                eKL( :, iDT, iDA ) = NaN;
            end
        end
    end
    toc
end


if ifPlotProb

    nTPlt = numel( idxTPlt );
    nYPlt = numel( idxYPlt );
    nTickSkip = 12;

    Fig.figWidth   = 6;    % in inches
    Fig.deltaX     = .55;
    Fig.deltaX2    = .85;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .3;
    Fig.gapX       = .3;
    Fig.gapY       = .6;
    Fig.nSkip      = 5;
    Fig.cLimScl    = 1;

    nTileX = nYPlt;
    nTileY = nTPlt;

    panelX = ( Fig.figWidth - Fig.deltaX - Fig.deltaX2 - ( nTileX -1 ) * Fig.gapX ) / nTileX;
    panelY = panelX * ( 3 / 4 ) ^3;

    posn     = [ 0 ...
                 0 ...
                 nTileX * panelX + ( nTileX - 1 ) * Fig.gapX + Fig.deltaX + Fig.deltaX2 ...
                 nTileY * panelY + ( nTileY - 1 ) * Fig.gapY + Fig.deltaY + Fig.deltaY2 ];


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
                    [ Fig.deltaX + ( iAx - 1 ) * ( panelX + Fig.gapX ), ...
                      Fig.deltaY + ( nTileY - jAx ) * ( panelY + Fig.gapY ), ...
                      panelX, panelY ] );
        end
    end

    axPos = reshape( get( ax, 'position' ), [ nTileX nTileY ] );


    for iY = 1 : nYPlt

        % bin coordinates
        aPlt = aY( idxYPlt( iY ), : );
        aPlt( 1 ) = 1.3 *  min( y( idxYPlt( iY ), : ) );
        aPlt( end ) = 1.3 * max( y( idxYPlt( iY ), : ) );
        daPlt = ( aPlt( 2 : end ) - aPlt( 1 : end - 1 ) )';

        for iT = 1 : nTPlt 

            % verification time serial numbers
            tPlt = tNumVer( ( 1 : nDT ) + idxTPlt( iT ) - 1, : );
            tPlt = tPlt( : )';
            ifTruePlt = tNumOut >= tPlt( 1 ) & tNumOut <= tPlt( end );
            ifObsPlt = tNumObs >= tPlt( 1 ) & tNumObs <= tPlt( end ); 
            [ aGrd, tGrd ] = ndgrid( ( aPlt( 1 : end - 1 ) + aPlt( 2 : end ) ) / 2, tPlt ); 
            tLabels = cellstr( datestr( tPlt , 'mm/yy' ) ); 
        
            pPlt = real( squeeze( ...
                 p( :, idxYPlt( iY ), ( 1 : nDT ) + idxTPlt( iT ) - 1, : ) ) ); 
            pPlt = reshape( pPlt, [ nQ nDT * nDA ] ); 
            pPlt = bsxfun( @rdivide, pPlt, daPlt );


            set( gcf, 'currentAxes', ax( iY, iT ) )
            contourf( tGrd, aGrd, log10( pPlt ), 4 );
            hold on
            plot( tNumOut( ifTruePlt ), yO( idxYPlt( iY ), find( ifTruePlt ) ), 'r-', 'linewidth', 2 )
            plot( tNumObs( ifObsPlt ), yObs( idxYPlt( iY ), find( ifObsPlt ) ), 'r.', 'linewidth', 2 )

            set( gca, 'tickDir', 'out', 'tickLength', [ 0.005 0 ] )
            axis tight
            set( gca, 'xTick', tPlt( 1 : nTickSkip : end ), 'xTickLabel', tLabels( 1 : nTickSkip : end' ) )
            %set( ax, 'yLim', [ aY( iY, 1 ) aY( iY, end ) ], 'xLim', tLimPlt )
            %set( hP, 'edgeColor', 'none' )
            set( gca, 'cLim', [ -4 0 ] )
            hC = colorbar( 'location', 'eastOutside' );
            cPos = get( hC, 'position' );
            cPos( 3 ) = .5 * cPos( 3 );
            cPos( 1 ) = cPos( 1 ) + .11;
            set( hC, 'position', cPos );
            ylabel( hC, 'log_{10}P' )
            set( gca, 'position', axPos{ iY, iT } );
            title( sprintf( 'Lead time \\tau = %i months', idxTPlt( iT ) - 1 ) )
            if iY == 1 
                ylabel( 'Nino 3.4 index' )
            end
            if iT == nTPlt
                xlabel( 'verification time' )
            end

        end
        %text( -.08, 1.05, lbl, 'units', 'normalized' )
        %text( tPos( 1 ), tPos( 2 ), lbl )
        %title( sprintf( 'Gridpoint %i/%i', idxY( iY ), Pars.n ) )
        %title( [ lblStr sprintf( ' \\Delta t / T = %1.2g', nDT * dt / 2 / pi * omega ) ] )

    end
    print -dpng -r300 figNino34Prob.png
end



if ifPlotEnt

    nTPlt = numel( idxTPlt );
    nYPlt = numel( idxYPlt );
    nTickSkip = 12;

    Fig.figWidth   = 6;    % in inches
    Fig.deltaX     = .55;
    Fig.deltaX2    = .85;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .3;
    Fig.gapX       = .3;
    Fig.gapY       = .6;
    Fig.nSkip      = 5;
    Fig.cLimScl    = 1;

    nTileX = nYPlt;
    nTileY = nTPlt;

    panelX = ( Fig.figWidth - Fig.deltaX - Fig.deltaX2 - ( nTileX -1 ) * Fig.gapX ) / nTileX;
    panelY = panelX * ( 3 / 4 )^3;

    posn     = [ 0 ...
                 0 ...
                 nTileX * panelX + ( nTileX - 1 ) * Fig.gapX + Fig.deltaX + Fig.deltaX2 ...
                 nTileY * panelY + ( nTileY - 1 ) * Fig.gapY + Fig.deltaY + Fig.deltaY2 ];


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
                    [ Fig.deltaX + ( iAx - 1 ) * ( panelX + Fig.gapX ), ...
                      Fig.deltaY + ( nTileY - jAx ) * ( panelY + Fig.gapY ), ...
                      panelX, panelY ] );
        end
    end

    axPos = reshape( get( ax, 'position' ), [ nTileX nTileY ] );

    for iY = 1 : nYPlt
        for iT = 1 : nTPlt

            % verification time serial numbers
            tPlt = tNumVer( ( 1 : nDT ) + idxTPlt( iT ) - 1, : );
            tPlt = tPlt( : )';
            ifTruePlt = tPlt <= tNumOut( end ); 
            idxTObsPlt = find( tNumObs >= tPlt( 1 ) & tNumObs <= tPlt( end ) ); 
            tLabels = cellstr( datestr( tPlt , 'mm/yy' ) ); 

            dKLPlt = squeeze( ...
                 dKL( idxYPlt( iY ), ( 1 : nDT ) + idxTPlt( iT ) - 1, : ) ); 
            dKLPlt = dKLPlt( : );

            eKLPlt = squeeze( ...
                 eKL( idxYPlt( iY ), ( 1 : nDT ) + idxTPlt( iT ) - 1, : ) ); 
            eKLPlt = eKLPlt( : );

            set( gcf, 'currentAxes', ax( iY, iT ) )
            plot( tPlt( ifTruePlt ), eKLPlt( ifTruePlt ), 'g-' )
            hold on
            plot( tPlt, dKLPlt, 'b-' ) 
            plot( tPlt, sign( tPlt ) * log2( nQ ), 'm-' )
            axis tight
            %legend( 'E', 'D' ) 
            %legend boxoff
            yLm = get( gca, 'yLim' );
            for iObs = 1 : numel( idxTObsPlt )
                plot( tNumObs( idxTObsPlt( iObs ) ) * [ 1 1 ], yLm, 'r:' )
            end
            title( sprintf( 'Lead time \\tau = %i months', idxTPlt( iT ) - 1 ) )
            if iY == 1 
                ylabel( 'Nino 3.4 index' )
            end
            if iT == nTPlt
                xlabel( 'verification time' )
            end
            set( ax, 'tickDir', 'out', 'tickLength', [ 0.005 0 ] )
            set( gca, 'xTick', tPlt( 1 : nTickSkip : end ), 'xTickLabel', tLabels( 1 : nTickSkip : end' ) )
        end
    end

    print -dpng -r300 figNino34Ent.png
end

if ifMovieProb


    nTPlt = numel( idxTPlt );
    nYPlt = numel( idxYPlt );
    
    % bin coordinates
    aPlt = cell( 1, nYPlt );
    daPlt = cell( 1, nYPlt );
    for iY = 1 : nYPlt 
        aPlt{ iY } = aY( idxYPlt( iY ), : ); 
        aPlt{ iY }( 1 ) = min( y( idxYPlt( iY ), : ), [], 2 );
        aPlt{ iY }( end ) = max( y( idxYPlt( iY ), : ), [], 2 );
        daPlt{ iY } = ( aPlt{ iY }( 2 : end ) - aPlt{ iY }( 1 : end - 1 ) )';
    end

    % verification time serial numbers
    tPlt = cell( 1, nTPlt );
    ifTruePlt = cell( 1, nTPlt );
    ifObsPlt = cell( 1, nTPlt );
    tLabels = cell( 1, nTPlt );
    for iT = 1 : nTPlt
        tPlt{ iT } = tNumVer( ( 1 : nDT ) + idxTPlt( iT ) - 1, : );
        tPlt{ iT } = tPlt{ iT }( : )';
        ifTruePlt{ iT } = tNumOut >= tPlt{ iT }( 1 ) ...
                        & tNumOut <= tPlt{ iT }( end );
        ifObsPlt{ iT } = tNumObs >= tPlt{ iT }( 1 ) ...
                       & tNumObs <= tPlt{ iT }( end ); 
        tLabels{ iT } = cellstr( datestr( tPlt{ iT } , 'mm/yy' ) ); 
    end
    nFrame = numel( tPlt{ 1 } );
    
    % probability values
    pPlt = cell( nTPlt, nYPlt );
    pLim = zeros( nTPlt, nYPlt );
    for iY = 1 : nYPlt
       for iT = 1 : nTPlt 
            pPlt{ iT, iY } = real( squeeze( ...
               p( :, idxYPlt( iY ), ( 1 : nDT ) + idxTPlt( iT ) - 1, : ) ) ); 
            pPlt{ iT, iY } = reshape( pPlt{ iT, iY }, [ nQ nDT * nDA ] ); 
            pPlt{ iT, iY } = bsxfun( @rdivide, pPlt{ iT, iY }, daPlt{ iY } );
            pLim( iT, iY ) = max( pPlt{ iT, iY }( : ) );
        end
    end
        
    Mov.figWidth   = 300;    % in pixels
    Mov.deltaX     = 35;
    Mov.deltaX2    = 15;
    Mov.deltaY     = 35;
    Mov.deltaY2    = 40;
    Mov.gapX       = 30;
    Mov.gapY       = 40;
    Mov.visible    = 'on';
    Mov.fps        = 12;

    nTileX = nYPlt;
    nTileY = nTPlt;

    panelX = ( Mov.figWidth - Mov.deltaX - Mov.deltaX2 - ( nTileX -1 ) * Mov.gapX ) / nTileX;
    panelY = panelX * 3 / 4;

    posn     = [ 0, ...
                 0, ...
                 nTileX * panelX + ( nTileX - 1 ) * Mov.gapX + Mov.deltaX + Mov.deltaX2, ...
                 nTileY * panelY + ( nTileY - 1 ) * Mov.gapY + Mov.deltaY + Mov.deltaY2 ];

    writerObj = VideoWriter( 'movieNino34_density.avi' );
    writerObj.FrameRate = Mov.fps;
    writerObj.Quality = 100;
    open( writerObj );

    fig = figure( 'units', 'pixels', ...
              'paperunits', 'points', ...
              'position', posn, ...
              'paperPosition', posn, ...
              'visible', Mov.visible, ...
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
              'defaultAxesLineWidth', 1, ...
              'defaultAxesLayer', 'top' );

    ax = zeros( nTileX, nTileY );

    for iAx = 1 : nTileX
        for jAx = 1 : nTileY
            ax( iAx, jAx ) = axes( ...
                'units', 'pixels', ...
                'position', ...
                [ Mov.deltaX + ( iAx - 1 ) * ( panelX + Mov.gapX ), ...
                  Mov.deltaY + ( nTileY - jAx ) * ( panelY + Mov.gapY ), ...
                  panelX, panelY ] );
        end
    end

    axTitle = axes( 'units', 'pixels', 'position', [ Mov.deltaX, Mov.deltaY, ...
                              nTileX * panelX + ( nTileX - 1 ) * Mov.gapX, ...
                              nTileY * panelY + ( nTileY - 1 ) * Mov.gapY + 15 ], ...
                    'color', 'none', 'box', 'off' );

    for iFrame = 1 : nFrame
        for iY = 1 : nYPlt 
            for iT = 1 : nTPlt 
                set( gcf, 'currentAxes', ax( iY, iT ) ) 
                if iFrame - idxTPlt( iT ) >= 0    
                    histogram( 'binEdges', aPlt{ iY }, ...
                               'binCounts', pPlt{ iT, iY }( :, iFrame - idxTPlt( iT ) + 1 ) );
                    hold on
                    if ifTruePlt{ iT }( iFrame )
                        plot( yO( idxYPlt( iY ), iFrame ) * [ 1 1 ], ...
                              [ 0 1.1 ] * pLim( iT, iY ), 'r-' )
                    end
                    if ifObsPlt{ iT }( iFrame )
                        plot( yO( idxYPlt( iY ), iFrame ), ...
                              1, 'r*' )
                    end

                    set( gca, 'xLim', [ aPlt{ iY }( 1 ) - 0.1 * ( aPlt{ iY }( end ) - aPlt{ iY }( 1 ) ), ...
                    aPlt{ iY }( end ) + 0.1 * ( aPlt{ iY }( end ) - aPlt{ iY }( 1 ) ) ], ... 
                    'yLim', [ 0 1 ] )  
                else
                    axis off
                end
                title( sprintf( 'Lead time %i months', idxTPlt( iT ) - 1 ) ) 
            end
        end
        
        title( axTitle, [ 'Verification time ' tLabels{ 1 }{ iFrame } ] )
        axis( axTitle, 'off' )

        frame = getframe( gcf );
        writeVideo( writerObj, frame )

        for iY = 1 : nYPlt
            for iT = 1 : nTPlt
                cla( ax( iY, iT ), 'reset' )
            end
        end
        cla( axTitle, 'reset' )
    end

    close( writerObj )
end
