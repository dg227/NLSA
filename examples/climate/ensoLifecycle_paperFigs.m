modelExperiment = 'ccsm4Ctrl_1300yr_IPSST_4yrEmb_coneKernel';
obsExperiment = 'ersstV4_50yr_IPSST_4yrEmb_coneKernel';

ifSpectrum = false; 
ifCompositesM = false;
ifCompositesO = true;

figDir = 'paperFigs';
if ~isdir( figDir )
    mkdir( figDir )
end

modelM = ensoLifecycle_nlsaModel( modelExperiment );
modelO = ensoLifecycle_nlsaModel( obsExperiment ); 

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


