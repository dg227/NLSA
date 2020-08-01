modelExperiment = 'ccsm4Ctrl_1300yr_IPSST_4yrEmb_coneKernel';
obsExperiment = 'ersstV4_50yr_IPSST_4yrEmb_coneKernel';

ifSpectrum = true; 

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
                  'triennial' ...
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
    Fig.deltaX     = .5;
    Fig.deltaX2    = 2.1;
    Fig.deltaY     = .48;
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
    ylabel( 'frequency (1/y)' )
    xlabel( 'decay rate (arbitrary units)' )
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
    xlabel( 'decay rate (arbitrary units)' )
    grid on
    title( '(b) ERSSTv4 spectrum' )

    figFile = fullfile( figDir, 'figGeneratorSpectrum.png' );
    print( fig, figFile, '-dpng', '-r300' ) 


end
