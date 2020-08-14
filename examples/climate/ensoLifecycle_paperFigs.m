modelExperiment = 'ccsm4Ctrl_1300yr_IPSST_4yrEmb_coneKernel';
obsExperiment = 'ersstV4_50yr_IPSST_4yrEmb_coneKernel';

ifSpectrum = false; 
ifLifecycleM = true;
ifLifecycleO = true;
ifCompositesM = false;
ifCompositesO = false;

figDir = 'paperFigs';
if ~isdir( figDir )
    mkdir( figDir )
end

modelM = ensoLifecycle_nlsaModel( modelExperiment );
modelO = ensoLifecycle_nlsaModel( obsExperiment ); 

ElNinos = { { '201511' '201603' } ... 
            { '199711' '199803' } ...
            { '199111' '199203' } ...
            { '198711' '198803' } ...
            { '198211' '198303' } ...
            { '197211' '197303' } ...
            { '196511' '196603' } ...
            { '195711' '195803' } };

LaNinas = { { '201011' '201103' } ... 
            { '200711' '200803' } ...
            { '199911' '200003' } ...
            { '199811' '199903' } ...
            { '198811' '198903' } ...
            { '197511' '197603' } ...
            { '197311' '197403' } };





if ifSpectrum

    gammaM = getKoopmanEigenvalues( modelM ) * 12 / 2 / pi; 
    gammaO = getKoopmanEigenvalues( modelO ) * 12 / 2 / pi; 

    legendStr = { 'constant' ...
                  'annual' ...
                  'semiannual' ...
                  'triannual' ...
                  'trend' ... 
                  'trend combination' ...
                  'ENSO' ...
                  'ENSO combination' ...
                  'decadal' ...
                  'seasonal branch' ...
                  'trend branch' ...
                  'ENSO branch' };

    %c = distinguishable_colors( 9 );
    %SpecColor.const = c( 1, : );
    %SpecColor.annual = c( 2, : );
    %SpecColor.semi = c( 3, : );
    %SpecColor.tri = c( 4, : );
    %SpecColor.trend = c( 5, : );
    %SpecColor.trendC = c( 6, : );
    %SpecColor.enso = c( 7, : );
    %SpecColor.ensoC = c( 8, : );
    %SpecColor.decadal = c( 9, : );

    SpecColor.const = [ 0 0 0 ];
    SpecColor.annual = [ 0 0 1 ];
    SpecColor.semi = [ 1 1 0 ] * .75;
    SpecColor.tri = [ 1 1 0 ];
    SpecColor.trend = [ 0 1 0 ];
    SpecColor.trendC = [ 0 1 1 ];
    SpecColor.enso = [ 1 0 0 ];
    SpecColor.ensoC = [ 1 0 1 ];
    SpecColor.decadal = [ 1 1 0 ] * .5;

    obsSpec.const = 1;
    obsSpec.annual = [ 2 3 ];
    obsSpec.semi   = [ 4 5 ];
    obsSpec.tri = [ 13 14 ];
    obsSpec.trend = 6;
    obsSpec.trendC = [ 9 10 ];
    obsSpec.enso = [ 7 8 ];
    obsSpec.ensoC = [ 11 12 16 17];
    obsSpec.decadal = 15;
    obsSpec.annualBranch = [ 13 4 2 3 5 14 ];
    obsSpec.trendBranch = [ 9 6 10 ];
    obsSpec.ensoBranch = [ 16 11 7 8 12 17 ];
    obsSpec.ifMark = false( size( gammaO ) );
    obsSpec.ifMark( [ obsSpec.const obsSpec.annual obsSpec.semi ...
                        obsSpec.tri obsSpec.enso obsSpec.decadal ...
                        obsSpec.trend obsSpec.trendC ] ) = true;

    modelSpec.const = 1;
    modelSpec.annual = [ 2 3 ];
    modelSpec.semi   = [ 4 5 ];
    modelSpec.tri = [ 6 7 ];
    modelSpec.enso = [ 8 9 ];
    modelSpec.ensoC = [ 10 : 17];
    modelSpec.decadal = 29;
    modelSpec.annualBranch = [ 6 4 2 3 5 7 ];
    modelSpec.ensoBranch = [ 16 14 12 10 8 9 11 13 15 17 ];
    modelSpec.ifMark = false( size( gammaM ) );
    modelSpec.ifMark( [ modelSpec.const modelSpec.annual modelSpec.semi ...
                        modelSpec.tri modelSpec.enso modelSpec.decadal ] ) = true;


    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 8; 
    Fig.deltaX     = .55;
    Fig.deltaX2    = 2.1;
    Fig.deltaY     = .55;
    Fig.deltaY2    = .2;
    Fig.gapX       = .25;
    Fig.gapY       = .5;
    Fig.gapT       = 0; 
    Fig.nTileX     = 2;
    Fig.nTileY     = 1;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 6;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );



    set( gcf, 'currentAxes', ax( 1 ) )
    plot( real( gammaM( modelSpec.const ) ), ...
          imag( gammaM( modelSpec.const ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.const )
    plot( real( gammaM( modelSpec.annual ) ), ...
          imag( gammaM( modelSpec.annual ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.annual )
    plot( real( gammaM( modelSpec.semi ) ), ...
          imag( gammaM( modelSpec.semi ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.semi )
    plot( real( gammaM( modelSpec.tri ) ), ...
          imag( gammaM( modelSpec.tri ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.tri )
    plot( real( gammaM( modelSpec.enso ) ), ...
          imag( gammaM( modelSpec.enso ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.enso )
    plot( real( gammaM( modelSpec.ensoC ) ), ...
          imag( gammaM( modelSpec.ensoC ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.ensoC )
    plot( real( gammaM( modelSpec.decadal ) ), ...
          imag( gammaM( modelSpec.decadal ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.decadal )
    plot( real( gammaM( modelSpec.annualBranch ) ), ...
          imag( gammaM( modelSpec.annualBranch ) ), 'b-' )
    plot( real( gammaM( modelSpec.ensoBranch ) ), ...
          imag( gammaM( modelSpec.ensoBranch ) ), 'r-' )
    plot( real( gammaM( ~modelSpec.ifMark ) ), ...
          imag( gammaM( ~modelSpec.ifMark ) ), '.', 'markerSize', 10, ...
          'color', [ .5 .5 .5 ] )
    xlim( [ -4 .1 ] );
    ylim( [ -3 3 ] ); 
    ylabel( 'Frequency \nu_j (cycles/y)' )
    xlabel( 'Decay rate Re(\gamma_j) (arbitrary units)' )
    grid on
    title( '(a) CCSM4 spectrum' )

    set( gcf, 'currentAxes', ax( 2, 1 ) )
    set( gca, 'yTickLabel', [] )
    plot( real( gammaO( obsSpec.const ) ), ...
          imag( gammaO( obsSpec.const ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.const )
    plot( real( gammaO( obsSpec.annual ) ), ...
          imag( gammaO( obsSpec.annual ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.annual )
    plot( real( gammaO( obsSpec.semi ) ), ...
          imag( gammaO( obsSpec.semi ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.semi )
    plot( real( gammaO( obsSpec.tri ) ), ...
          imag( gammaO( obsSpec.tri ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.tri )
    plot( real( gammaO( obsSpec.trend ) ), ...
          imag( gammaO( obsSpec.trend ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.trend )
    plot( real( gammaO( obsSpec.trendC ) ), ...
          imag( gammaO( obsSpec.trendC ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.trendC )
    plot( real( gammaO( obsSpec.enso ) ), ...
          imag( gammaO( obsSpec.enso ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.enso )
    plot( real( gammaO( obsSpec.ensoC ) ), ...
          imag( gammaO( obsSpec.ensoC ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.ensoC )
    plot( real( gammaO( obsSpec.decadal ) ), ...
          imag( gammaO( obsSpec.decadal ) ), ...
          '.', 'markersize', 15, 'color', SpecColor.decadal )
    plot( real( gammaO( obsSpec.annualBranch ) ), ...
          imag( gammaO( obsSpec.annualBranch ) ), 'b-' )
    plot( real( gammaO( obsSpec.trendBranch ) ), ...
          imag( gammaO( obsSpec.trendBranch ) ), 'g-' )
    plot( real( gammaO( obsSpec.ensoBranch ) ), ...
          imag( gammaO( obsSpec.ensoBranch ) ), 'r-' )
    plot( real( gammaO( ~obsSpec.ifMark ) ), ...
          imag( gammaO( ~obsSpec.ifMark ) ), '.', 'markerSize', 10, ...
          'color', [ .5 .5 .5 ] )
    axPos = get( gca, 'position' );
    hL = legend( legendStr{ : }, 'location', 'eastOutside' );
    set( gca, 'position', axPos )
    lPos = get( hL, 'position' );
    lPos( 1 ) = lPos( 1 ) + .007; 
    set( hL, 'position', lPos );
    xlim( [ -1 .1 ] );
    ylim( [ -3 3 ] ); 
    xlabel( 'Decay rate Re(\gamma_j) (arbitrary units)' )
    grid on
    title( '(b) ERSSTv4 spectrum' )

    figFile = fullfile( figDir, 'figGeneratorSpectrum.png' );
    print( fig, figFile, '-dpng', '-r300' ) 


end


if ifLifecycleM

    model = modelM;
    idxZEnso     = 8;         
    phaseZ       = exp( i * pi * ( 17 / 32 + 1 ) );        
    iCNino34 = 1;  % Nino 3.4 index
    nSE          = getNTotalSample( model.embComponent );
    nSB          = getNXB( model.embComponent );
    nShiftTakens = floor( getEmbeddingWindow( model.embComponent ) / 2 );
    nShiftNino   = 11;        
    decayFactor  = 4; 
    nSamplePhase = 200;
    nPhase = 8;

    idxTLim = [ 1 1200 ] + 9000;
    idxTLimTS = [ 1 361 ] + 8999;
    idxTick = idxTLimTS( 1 ) : 60 : idxTLimTS( 2 );

    % Retrieve Koopman eigenfunctions
    z = getKoopmanEigenfunctions( model );
    T = getEigenperiods( model.koopmanOp );
    TEnso = abs( T( idxZEnso ) / 12 );
    Z.idx = [ real( phaseZ * z( :, idxZEnso ) )' 
             imag( phaseZ * z( :, idxZEnso ) )' ];
    Z.time = getTrgTime( model );
    Z.time = Z.time( nSB + 1 + nShiftTakens : end );
    Z.time = Z.time( 1 : nSE );
    
    % Construct lagged Nino 3.4 indices
    Nino34.time = getTrgTime( model ); 
    Nino34.time = Nino34.time( nSB + 1 + nShiftTakens : end );
    Nino34.time = Nino34.time( 1 : nSE );

    % Nino 3.4 index
    nino = getData( model.trgComponent( iCNino34 ) );
    Nino34.idx = [ nino( nShiftNino + 1 : end ) 
                 nino( 1 : end - nShiftNino ) ];
    Nino34.idx = Nino34.idx( :, nSB + nShiftTakens - nShiftNino + 1 : end );
    Nino34.idx = Nino34.idx( :, 1 : nSE );

    [ Z.selectInd, Z.angles, Z.avNino34, Z.weights ] = ...
        computeLifecyclePhasesWeighted( Z.idx', Nino34.idx( 1, : )', ...
                                        nPhase, nSamplePhase, decayFactor );
    [ Nino34.selectInd, Nino34.angles, Nino34.avNino34, Nino34.weights ] = ...
        computeLifecyclePhasesWeighted( Nino34.idx', Nino34.idx( 1, : )', ...
                                        nPhase, nSamplePhase, decayFactor );

    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 12; 
    Fig.deltaX     = .8;
    Fig.deltaX2    = 1.2;
    Fig.deltaY     = .6;
    Fig.deltaY2    = .3;
    Fig.gapX       = .30;
    Fig.gapY       = .7;
    Fig.gapT       = .3; 
    Fig.nTileX     = 3;
    Fig.nTileY     = 2;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 13;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    % Plot Nino 3.4 lifecycle
    set( gcf, 'currentAxes', ax( 1, 1 ) )
    plotLifecycle( Nino34, ElNinos, LaNinas, model.tFormat, idxTLim )
    xlabel( 'Nino 3.4' )
    ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( [ -4 4 ] )
    ylim( [ -4 4 ] )
    set( gca, 'xTick', -4 : 1 : 4, 'yTick', -4 : 1 : 4 )
    title( 'Nino 3.4' )

    % Plot generator lifecycle
    set( gcf, 'currentAxes', ax( 2, 1 ) )
    plotLifecycle( Z, ElNinos, LaNinas, model.tFormat, idxTLim )
    xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    ylabel( sprintf( 'Im(z_{%i})', idxZEnso ) )
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    title( sprintf( 'Generator; eigenperiod = %1.2f y', TEnso ) )
    set( gca, 'yAxisLocation', 'right' )

    % Plot Nino 3.4 time series
    set( gcf, 'currentAxes', ax( 3, 1 ) )
    plot( Nino34.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
          Nino34.idx( 1, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), 'b-' )
    ylim( [ -4 4 ] )
    xlim( [ Nino34.time( idxTLimTS( 1 ) ) Nino34.time( idxTLimTS( 2 ) ) ] ) 
    set( gca, 'xTick', Nino34.time( idxTick ), ...
              'xTickLabel', datestr( Nino34.time( idxTick ), 'mm/yy' ), ...
              'yAxisLocation', 'right' ) 
    grid on
    title( 'Time series' )
    legend( 'Nino 3.4', 'location', 'southWest' )
    axPos = get( gca, 'position' );
    axPos( 1 ) = axPos( 1 ) + .7;
    set( gca, 'position', axPos )
    
    % Make scatterplot of Nino 3.4 lifecycle colored by Nino 3.4 index
    set( gcf, 'currentAxes', ax( 1, 2 ) )
    plot( Nino34.idx( 1, : ), Nino34.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Nino34.idx( 1, : ), Nino34.idx( 2, : ), 17, Nino34.idx( 1, : ), ...
             'o', 'filled' )  
    %xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    xlim( [ -4 4 ] )
    ylim( [ -4 4 ] )
    set( gca, 'clim', [ -3 3 ], 'xTick', [ -4 : 4 ], 'yTick', [ -4 : 4 ], ...
         'yTickLabel', [] )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )
    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'westOutside' );
    cPos = get( hC, 'position' );
    cPos( 3 ) = cPos( 3 ) * .7;
    cPos( 1 ) = cPos( 1 ) - .05;
    set( hC, 'position', cPos )
    xlabel( hC, 'Nino 3.4 index' )
    set( gca, 'position', axPos )
    [ minNina, idxMinNina ] = min( Nino34.avNino34 ); 
    angleRef = Nino34.angles( idxMinNina );
    indRef = ( 0 : .1 : 4 ) * exp( i * angleRef );
    plot( indRef, 'g--', 'lineWidth', 2 )
    %xlabel( 'Nino 3.4' )

    % Make scatterplot of generator lifecycle colored by Nino 3.4 index
    set( gcf, 'currentAxes', ax( 2, 2 ) )
    scl = max( abs( Nino34.idx( 1, : ) ) );
    plot( Z.idx( 1, : ), Z.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Z.idx( 1, : ), Z.idx( 2, : ), 17, Nino34.idx( 1, : ) / scl, ...
             'o', 'filled' )  
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    set( gca, 'clim', [ -1 1 ] )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )
    set( gca, 'xTick', [ -2 : 1 : 2 ], 'yTick', [ -2 : 1 : 2 ], ...
              'yAxisLocation', 'right' );
    [ minNina, idxMinNina ] = min( Z.avNino34 ); 
    angleRef = Z.angles( idxMinNina );
    indRef = ( 0 : .1 : 4 ) * exp( i * angleRef );
    plot( indRef, 'g--', 'lineWidth', 2 )

    % Plot generator time series
    set( gcf, 'currentAxes', ax( 3, 2 ) )
    plot( Z.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
          Z.idx( 1, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), 'b-' )
    %hold on
    %plot( Z.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
    %      Z.idx( 2, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), 'r-' )
    ylim( [ -2.5 2.5 ] )
    set( gca, 'xTick', Z.time( idxTick ), ...
              'xTickLabel', datestr( Z.time( idxTick ), 'mm/yy' ) ) 
    set( gca, 'yAxisLocation', 'right' )  
    grid on
    legend( sprintf( 'Re(z_{%i})', idxZEnso ), ...
            'location', 'southWest' )
    axPos = get( gca, 'position' );
    axPos( 1 ) = axPos( 1 ) + .7;
    set( gca, 'position', axPos )
    xlim( [ Z.time( idxTLimTS( 1 ) ) Z.time( idxTLimTS( 2 ) ) ] ) 

    title( axTitle, '(a) CCSM4 ENSO lifecycle' );

    set( gcf, 'invertHardCopy', 'off' )
    figFile = fullfile( figDir, 'figEnsoLifecycleGenerator_ccsm4.png' );
    print( fig, figFile, '-dpng', '-r300' ) 

end
 
if ifLifecycleO

    model = modelO;
    idxZEnso     = 7;         
    phaseZ       = exp( i * 5 * pi / 32 );        
    iCNino34 = 1;  % Nino 3.4 index
    nSE          = getNTotalSample( model.embComponent );
    nSB          = getNXB( model.embComponent );
    nShiftTakens = floor( getEmbeddingWindow( model.embComponent ) / 2 );
    nShiftNino   = 11;        

    idxTLim = [ 1 nSE ];
    idxTLimTS = [ nSE - 360, nSE ];
    idxTick = idxTLimTS( 1 ) : 60 : idxTLimTS( 2 );

    decayFactor  = 4; 
    nSamplePhase = 200;
    nPhase = 8;

    % Retrieve Koopman eigenfunctions
    z = getKoopmanEigenfunctions( model );
    T = getEigenperiods( model.koopmanOp );
    TEnso = abs( T( idxZEnso ) / 12 );
    Z.idx = [ real( phaseZ * z( :, idxZEnso ) )' 
             imag( phaseZ * z( :, idxZEnso ) )' ];
    Z.time = getTrgTime( model );
    Z.time = Z.time( nSB + 1 + nShiftTakens : end );
    Z.time = Z.time( 1 : nSE );
    
    % Construct lagged Nino 3.4 indices
    Nino34.time = getTrgTime( model ); 
    Nino34.time = Nino34.time( nSB + 1 + nShiftTakens : end );
    Nino34.time = Nino34.time( 1 : nSE );

    % Nino 3.4 index
    nino = getData( model.trgComponent( iCNino34 ) );
    Nino34.idx = [ nino( nShiftNino + 1 : end ) 
                 nino( 1 : end - nShiftNino ) ];
    Nino34.idx = Nino34.idx( :, nSB + nShiftTakens - nShiftNino + 1 : end );
    Nino34.idx = Nino34.idx( :, 1 : nSE );

    
    [ Z.selectInd, Z.angles, Z.avNino34, Z.weights ] = ...
        computeLifecyclePhasesWeighted( Z.idx', Nino34.idx( 1, : )', ...
                                        nPhase, nSamplePhase, decayFactor );
    [ Nino34.selectInd, Nino34.angles, Nino34.avNino34, Nino34.weights ] = ...
        computeLifecyclePhasesWeighted( Nino34.idx', Nino34.idx( 1, : )', ...
                                        nPhase, nSamplePhase, decayFactor );

    % Set up figure and axes 
    Fig.units      = 'inches';
    Fig.figWidth   = 12; 
    Fig.deltaX     = .8;
    Fig.deltaX2    = 1.2;
    Fig.deltaY     = .6;
    Fig.deltaY2    = .3;
    Fig.gapX       = .30;
    Fig.gapY       = .7;
    Fig.gapT       = .3; 
    Fig.nTileX     = 3;
    Fig.nTileY     = 2;
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 13;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );

    % Plot Nino 3.4 lifecycle
    set( gcf, 'currentAxes', ax( 1, 1 ) )
    plotLifecycle( Nino34, ElNinos, LaNinas, model.tFormat, idxTLim )
    xlabel( 'Nino 3.4' )
    ylabel( sprintf( 'Nino 3.4 - %i months', nShiftNino ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    set( gca, 'xTick', -3 : 1 : 3, 'yTick', -3 : 1 : 3 )
    title( 'Nino 3.4' )

    % Plot generator lifecycle
    set( gcf, 'currentAxes', ax( 2, 1 ) )
    plotLifecycle( Z, ElNinos, LaNinas, model.tFormat, idxTLim )
    xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    ylabel( sprintf( 'Im(z_{%i})', idxZEnso ) )
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    title( sprintf( 'Generator; eigenperiod = %1.2f y', TEnso ) )
    set( gca, 'yAxisLocation', 'right' )

    % Plot Nino 3.4 time series
    set( gcf, 'currentAxes', ax( 3, 1 ) )
    plot( Nino34.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
          Nino34.idx( 1, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), 'b-' )
    ylim( [ -3 3 ] )
    xlim( [ Nino34.time( idxTLimTS( 1 ) ) Nino34.time( idxTLimTS( 2 ) ) ] ) 
    set( gca, 'xTick', Nino34.time( idxTick ), ...
              'xTickLabel', datestr( Nino34.time( idxTick ), 'mm/yy' ), ...
              'yAxisLocation', 'right' ) 
    grid on
    title( 'Time series' )
    legend( 'Nino 3.4', 'location', 'southWest' )
    axPos = get( gca, 'position' );
    axPos( 1 ) = axPos( 1 ) + .7;
    set( gca, 'position', axPos )
    
    % Make scatterplot of Nino 3.4 lifecycle colored by Nino 3.4 index
    set( gcf, 'currentAxes', ax( 1, 2 ) )
    plot( Nino34.idx( 1, : ), Nino34.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Nino34.idx( 1, : ), Nino34.idx( 2, : ), 17, Nino34.idx( 1, : ), ...
             'o', 'filled' )  
    %xlabel( sprintf( 'Re(z_{%i})', idxZEnso ) )
    xlim( [ -3 3 ] )
    ylim( [ -3 3 ] )
    set( gca, 'clim', [ -3 3 ], 'xTick', [ -4 : 4 ], 'yTick', [ -4 : 4 ], ...
         'yTickLabel', [] )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )
    axPos = get( gca, 'position' );
    hC = colorbar( 'location', 'westOutside' );
    cPos = get( hC, 'position' );
    cPos( 3 ) = cPos( 3 ) * .7;
    cPos( 1 ) = cPos( 1 ) - .05;
    set( hC, 'position', cPos )
    xlabel( hC, 'Nino 3.4 index' )
    set( gca, 'position', axPos )
    %xlabel( 'Nino 3.4' )
    [ minNina, idxMinNina ] = min( Nino34.avNino34 ); 
    angleRef = Nino34.angles( idxMinNina );
    indRef = ( 0 : .1 : 4 ) * exp( i * angleRef );
    plot( indRef, 'g--', 'lineWidth', 2 )

    % Make scatterplot of generator lifecycle colored by Nino 3.4 index
    set( gcf, 'currentAxes', ax( 2, 2 ) )
    scl = max( abs( Nino34.idx( 1, : ) ) );
    plot( Z.idx( 1, : ), Z.idx( 2, : ), '-', 'color', [ 0 .3 0 ] )
    scatter( Z.idx( 1, : ), Z.idx( 2, : ), 17, Nino34.idx( 1, : ) / scl, ...
             'o', 'filled' )  
    xlim( [ -2.5 2.5 ] )
    ylim( [ -2.5 2.5 ] )
    set( gca, 'clim', [ -1 1 ] )
    colormap( redblue )
    set( gca, 'color', [ 1 1 1 ] * .3 )
    set( gca, 'xTick', [ -2 : 1 : 2 ], 'yTick', [ -2 : 1 : 2 ], ...
              'yAxisLocation', 'right' );
    [ minNina, idxMinNina ] = min( Z.avNino34 ); 
    angleRef = Z.angles( idxMinNina );
    indRef = ( 0 : .1 : 4 ) * exp( i * angleRef );
    plot( indRef, 'g--', 'lineWidth', 2 )

    % Plot generator time series
    set( gcf, 'currentAxes', ax( 3, 2 ) )
    plot( Z.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
          Z.idx( 1, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), 'b-' )
    %hold on
    %plot( Z.time( idxTLimTS( 1 ) : idxTLimTS( 2 ) ), ...
    %      Z.idx( 2, idxTLimTS( 1 ) : idxTLimTS( 2 ) ), 'r-' )
    ylim( [ -2.5 2.5 ] )
    set( gca, 'xTick', Z.time( idxTick ), ...
              'xTickLabel', datestr( Z.time( idxTick ), 'mm/yy' ) ) 
    set( gca, 'yAxisLocation', 'right' )  
    grid on
    legend( sprintf( 'Re(z_{%i})', idxZEnso ), ...
            'location', 'southWest' )
    axPos = get( gca, 'position' );
    axPos( 1 ) = axPos( 1 ) + .7;
    set( gca, 'position', axPos )
    xlim( [ Z.time( idxTLimTS( 1 ) ) Z.time( idxTLimTS( 2 ) ) ] ) 

    title( axTitle, '(b) ERSSTv4 ENSO lifecycle' );

    set( gcf, 'invertHardCopy', 'off' )
    figFile = fullfile( figDir, 'figEnsoLifecycleGenerator_obs.png' );
    print( fig, figFile, '-dpng', '-r300' ) 

end
 



if ifCompositesM

    nPhase = 8;
    ifPlotWind = true;
    iCSST    = 5;  % global SST
    iCSAT    = 7;  % global SAT
    iCUWnd   = 9;  % global surface meridional winds
    iCVWnd   = 10; % global surface zonal winds

    dataFile = './figs/ccsm4Ctrl_1300yr_IPSST_4yrEmb_coneKernel/dataEnsoCompositesNino_globe_weighted.mat';
    load( dataFile )


    dataFile = './figs/ccsm4Ctrl_1300yr_IPSST_4yrEmb_coneKernel/dataEnsoCompositesGenerator_globe_weighted.mat';
    load( dataFile )
    
    SST.cOffset = SST.cOffset + .045;
    SAT.cOffset = SAT.cOffset + .055;
    SAT.cLim = [ -3 3];

    % SST phase composites

    % Figure and axes parameters 
    Fig.units      = 'inches';
    Fig.figWidth   = 6; 
    Fig.deltaX     = .55;
    Fig.deltaX2    = .5;
    Fig.deltaY     = .3;
    Fig.deltaY2    = .3;
    Fig.gapX       = .20;
    Fig.gapY       = .2;
    Fig.gapT       = .25; 
    Fig.nTileX     = 2;
    Fig.nTileY     = nPhase;
    Fig.aspectR    = (3/4)^3;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );
    colormap( redblue )

    for iPhase = 1 : nPhase

        SST.ifXTickLabels = iPhase == nPhase;
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        if ifPlotWind
            plotPhaseComposite( compNino34{ iCSST }( :, iPhase ), SST, ...
                                compNino34{ iCUWnd }( :, iPhase ), ...
                                compNino34{ iCVWnd }( :, iPhase ), UVWnd )
        else
            plotPhaseComposite( compNino34{ iCSST }( :, iPhase ), SST )
        end
        if iPhase == 1
            title( 'Nino 3.4' )
        end
        lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        lblPos = get( lbl, 'position' );
        lblPos( 1 ) = lblPos( 1 ) - .4;
        set( lbl, 'position', lblPos )

        set( fig, 'currentAxes', ax( 2, iPhase ) )
        SST.ifXTickLabels = iPhase == nPhase;
        SST.ifYTickLabels = false;
        if ifPlotWind
            plotPhaseComposite( compZ{ iCSST }( :, iPhase ), SST, ...
                                compZ{ iCUWnd }( :, iPhase ), ...
                                compZ{ iCVWnd }( :, iPhase ), UVWnd )
        else
            plotPhaseComposite( compZ{ iCSST }( :, iPhase ), SST )
        end
        SST.ifYTickLabels = true;

        if iPhase == 1
            title( 'Generator'  )
        end

    end


    titleStr = '(a) SST anomaly (K), surface wind';
    title( axTitle, titleStr )

    figFile = fullfile( figDir, 'figEnsoSSTComposites_ccsm4.png' );
    print( fig, figFile, '-dpng', '-r300' ) 

    % SAT phase composites

    % Figure and axes parameters 
    Fig.units      = 'inches';
    Fig.figWidth   = 6 - .6; 
    Fig.deltaX     = .55 -.6;
    Fig.deltaX2    = .5;
    Fig.deltaY     = .3;
    Fig.deltaY2    = .3;
    Fig.gapX       = .20;
    Fig.gapY       = .2;
    Fig.gapT       = .25; 
    Fig.nTileX     = 2;
    Fig.nTileY     = nPhase;
    Fig.aspectR    = (3/4)^3;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );
    colormap( redblue )

    for iPhase = 1 : nPhase

        SAT.ifYTickLabels = false;
        SAT.ifXTickLabels = iPhase == nPhase;
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        if ifPlotWind
            plotPhaseComposite( compNino34{ iCSAT }( :, iPhase ), SAT, ...
                                compNino34{ iCUWnd }( :, iPhase ), ...
                                compNino34{ iCVWnd }( :, iPhase ), UVWnd )
        else
            plotPhaseComposite( compNino34{ iCSAT }( :, iPhase ), SAT )
        end
        if iPhase == 1
            title( 'Nino 3.4' )
        end
        %lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        %lblPos = get( lbl, 'position' );
        %lblPos( 1 ) = lblPos( 1 ) - .4;
        %set( lbl, 'position', lblPos )

        set( fig, 'currentAxes', ax( 2, iPhase ) )
        SAT.ifXTickLabels = iPhase == nPhase;
        SAT.ifYTickLabels = false;
        if ifPlotWind
            plotPhaseComposite( compZ{ iCSAT }( :, iPhase ), SAT, ...
                                compZ{ iCUWnd }( :, iPhase ), ...
                                compZ{ iCVWnd }( :, iPhase ), UVWnd )
        else
            plotPhaseComposite( compZ{ iCSAT }( :, iPhase ), SAT )
        end
        %SAT.ifYTickLabels = true;

        if iPhase == 1
            title( 'Generator'  )
        end

    end


    titleStr = '(b) SAT anomaly (K), surface wind';
    title( axTitle, titleStr )

    figFile = fullfile( figDir, 'figEnsoSATComposites_ccsm4.png' );
    print( fig, figFile, '-dpng', '-r300' ) 


end


if ifCompositesO

    nPhase = 8;
    ifPlotWind = true;
    iCSST    = 5;  % global SST
    iCSAT    = 7;  % global SAT
    iCUWnd   = 9;  % global surface meridional winds
    iCVWnd   = 10; % global surface zonal winds

    dataFile = './figs/ersstV4_50yr_IPSST_4yrEmb_coneKernel/dataEnsoCompositesNino_globe_weighted.mat';
    load( dataFile )


    dataFile = './figs/ersstV4_50yr_IPSST_4yrEmb_coneKernel/dataEnsoCompositesGenerator_globe_weighted.mat';
    load( dataFile )
    
    SST.cOffset = SST.cOffset + .045;
    SAT.cOffset = SAT.cOffset + .055;
    SAT.cLim = [ -3 3];

    % SST phase composites

    % Figure and axes parameters 
    Fig.units      = 'inches';
    Fig.figWidth   = 6; 
    Fig.deltaX     = .55;
    Fig.deltaX2    = .5;
    Fig.deltaY     = .3;
    Fig.deltaY2    = .3;
    Fig.gapX       = .20;
    Fig.gapY       = .2;
    Fig.gapT       = .25; 
    Fig.nTileX     = 2;
    Fig.nTileY     = nPhase;
    Fig.aspectR    = (3/4)^3;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );
    colormap( redblue )

    for iPhase = 1 : nPhase

        SST.ifXTickLabels = iPhase == nPhase;
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        if ifPlotWind
            plotPhaseComposite( compNino34{ iCSST }( :, iPhase ), SST, ...
                                compNino34{ iCUWnd }( :, iPhase ), ...
                                compNino34{ iCVWnd }( :, iPhase ), UVWnd )
        else
            plotPhaseComposite( compNino34{ iCSST }( :, iPhase ), SST )
        end
        if iPhase == 1
            title( 'Nino 3.4' )
        end
        lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        lblPos = get( lbl, 'position' );
        lblPos( 1 ) = lblPos( 1 ) - .4;
        set( lbl, 'position', lblPos )

        set( fig, 'currentAxes', ax( 2, iPhase ) )
        SST.ifXTickLabels = iPhase == nPhase;
        SST.ifYTickLabels = false;
        if ifPlotWind
            plotPhaseComposite( compZ{ iCSST }( :, iPhase ), SST, ...
                                compZ{ iCUWnd }( :, iPhase ), ...
                                compZ{ iCVWnd }( :, iPhase ), UVWnd )
        else
            plotPhaseComposite( compZ{ iCSST }( :, iPhase ), SST )
        end
        SST.ifYTickLabels = true;

        if iPhase == 1
            title( 'Generator'  )
        end

    end


    titleStr = '(a) SST anomaly (K), surface wind';
    title( axTitle, titleStr )

    figFile = fullfile( figDir, 'figEnsoSSTComposites_obs.png' );
    print( fig, figFile, '-dpng', '-r300' ) 

    % SAT phase composites

    % Figure and axes parameters 
    Fig.units      = 'inches';
    Fig.figWidth   = 6 - .6; 
    Fig.deltaX     = .55 -.6;
    Fig.deltaX2    = .5;
    Fig.deltaY     = .3;
    Fig.deltaY2    = .3;
    Fig.gapX       = .20;
    Fig.gapY       = .2;
    Fig.gapT       = .25; 
    Fig.nTileX     = 2;
    Fig.nTileY     = nPhase;
    Fig.aspectR    = (3/4)^3;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 10;
    Fig.tickLength = [ 0.02 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax, axTitle ] = tileAxes( Fig );
    colormap( redblue )

    for iPhase = 1 : nPhase

        SAT.ifYTickLabels = false;
        SAT.ifXTickLabels = iPhase == nPhase;
        set( fig, 'currentAxes', ax( 1, iPhase ) )
        if ifPlotWind
            plotPhaseComposite( compNino34{ iCSAT }( :, iPhase ), SAT, ...
                                compNino34{ iCUWnd }( :, iPhase ), ...
                                compNino34{ iCVWnd }( :, iPhase ), UVWnd )
        else
            plotPhaseComposite( compNino34{ iCSAT }( :, iPhase ), SAT )
        end
        if iPhase == 1
            title( 'Nino 3.4' )
        end
        %lbl = ylabel(sprintf( 'Phase %i', iPhase ) );
        %lblPos = get( lbl, 'position' );
        %lblPos( 1 ) = lblPos( 1 ) - .4;
        %set( lbl, 'position', lblPos )

        set( fig, 'currentAxes', ax( 2, iPhase ) )
        SAT.ifXTickLabels = iPhase == nPhase;
        SAT.ifYTickLabels = false;
        if ifPlotWind
            plotPhaseComposite( compZ{ iCSAT }( :, iPhase ), SAT, ...
                                compZ{ iCUWnd }( :, iPhase ), ...
                                compZ{ iCVWnd }( :, iPhase ), UVWnd )
        else
            plotPhaseComposite( compZ{ iCSAT }( :, iPhase ), SAT )
        end
        %SAT.ifYTickLabels = true;

        if iPhase == 1
            title( 'Generator'  )
        end

    end


    titleStr = '(b) SAT anomaly (K), surface wind';
    title( axTitle, titleStr )

    figFile = fullfile( figDir, 'figEnsoSATComposites_obs.png' );
    print( fig, figFile, '-dpng', '-r300' ) 


end


function plotLifecycle( Index, Ninos, Ninas, tFormat, idxTLim )

if nargin < 5
    idxTLim = [ 1 size( Index.idx, 2 ) ];
end 


% plot temporal evolution of index
plot( Index.idx( 1, idxTLim( 1 ) : idxTLim( 2 ) ), Index.idx( 2, idxTLim( 1 ) : idxTLim( 2 ) ), 'g-' )
hold on
grid on

% highlight significant events
for iENSO = 1 : numel( Ninos )

    % Serial date numbers for start and end of event
    tLim = datenum( Ninos{ iENSO }( 1 : 2 ), tFormat );
    
    % Find and plot portion of index time series
    idxT1     = find( Index.time == tLim( 1 ) );
    idxT2     = find( Index.time == tLim( 2 ) );
    idxTLabel = round( ( idxT1 + idxT2 ) / 2 ); 
    plot( Index.idx( 1, idxT1 : idxT2 ), Index.idx( 2, idxT1 : idxT2 ), ...
          'r-', 'lineWidth', 2 )
    text( Index.idx( 1, idxTLabel ), Index.idx( 2, idxTLabel ), ...
          datestr( Index.time( idxT2 ), 'yyyy' ), 'fontSize', 8 )
end
for iENSO = 1 : numel( Ninas )

    % Serial date numbers for start and end of event
    tLim = datenum( Ninas{ iENSO }( 1 : 2 ), tFormat );
    
    % Find and plot portion of index time series
    idxT1 = find( Index.time == tLim( 1 ) );
    idxT2 = find( Index.time == tLim( 2 ) );
    idxTLabel = round( ( idxT1 + idxT2 ) / 2 ); 
    plot( Index.idx( 1, idxT1 : idxT2 ), Index.idx( 2, idxT1 : idxT2 ), ...
          'b-', 'lineWidth', 2 )
    text( Index.idx( 1, idxTLabel ), Index.idx( 2, idxTLabel ), ...
          datestr( Index.time( idxT2 ), 'yyyy' ), 'fontSize', 8 )
end

end





function plotPhaseComposite( s, SGrd, u, v, VGrd )

% s:    values of scalar field to plot
% SGrd: data structure with grid information for scalar field  
% u, v: components of vector field to plot
% VGrd: data structure with grid information for vector field

sData = zeros( size( SGrd.ifXY ) );
sData( ~SGrd.ifXY ) = NaN;
sData( SGrd.ifXY ) = s;
if isfield( SGrd, 'scl' )
    sData = SGrd.scl * sData; % apply scaling factor
end

if SGrd.ifXTickLabels
    xTickLabelsArg = { };
else
    xTickLabelsArg = { 'xTickLabels' [] };
end
if SGrd.ifYTickLabels
    yTickLabelsArg = { };
else
    yTickLabelsArg = { 'yTickLabels' [] };
end
m_proj( 'Miller cylindrical', 'long',  SGrd.xLim, 'lat', SGrd.yLim );
if ~isvector( SGrd.lon )
    SGrd.lon = SGrd.lon';
    SGrd.lat = SGrd.lat';
end
h = m_pcolor( SGrd.lon, SGrd.lat, sData' );
set( h, 'edgeColor', 'none' )
m_grid( 'linest', 'none', 'linewidth', 1, 'tickdir', 'out', ...
        xTickLabelsArg{ : }, yTickLabelsArg{ : } ); 
m_coast( 'linewidth', 1, 'color', 'k' );
        %'xTick', [ SGrd.xLim( 1 ) : 40 : SGrd.xLim( 2 ) ], ...
        %'yTick', [ SGrd.yLim( 1 ) : 20 : SGrd.yLim( 2 ) ] );

axPos = get( gca, 'position' );
hC = colorbar( 'location', 'eastOutside' );
cPos   = get( hC, 'position' );
cPos( 1 ) = cPos( 1 ) + SGrd.cOffset;
cPos( 3 ) = cPos( 3 ) * SGrd.cScale;
set( gca, 'cLim', SGrd.cLim, 'position', axPos )
set( hC, 'position', cPos )

if nargin == 2
    return
end

uData = zeros( size( VGrd.ifXY ) );
uData( ~VGrd.ifXY ) = NaN;
uData( VGrd.ifXY ) = u;

vData = zeros( size( VGrd.ifXY ) );
vData( ~VGrd.ifXY ) = NaN;
vData( VGrd.ifXY ) = v;

[ lon, lat ] = meshgrid( VGrd.lon, VGrd.lat );
%size(VGrd.lon)
%size(uData')
%size(vData')
m_quiver( lon( 1 : VGrd.nSkipY : end, 1 : VGrd.nSkipX : end ), ...
          lat( 1 : VGrd.nSkipY : end, 1 : VGrd.nSkipX : end ), ...
          uData( 1 : VGrd.nSkipX : end, 1 : VGrd.nSkipY : end )', ...
          vData( 1 : VGrd.nSkipX : end, 1 : VGrd.nSkipY : end )', ...
          'g-', 'lineWidth', .5 ) 
end


