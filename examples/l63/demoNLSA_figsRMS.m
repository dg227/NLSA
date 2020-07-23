experiments = { '64k_dt0.01_nEL0' '64k_dt0.01_nEL800' };
labels = { 'T = 0' 'T = 8' };
markers = { 'ro-' 'bo-' };
idxPhi = { [ 2 3 ] [ 2 3 ] };

nT = 1000;

ifPlotEigs = false;
ifPlotZ = false;
ifPlotAlpha = true;
ifPlotOmega = true;

figDir = 'figs_rms';

if ~isdir( figDir )
    mkdir( figDir )
end

if ifPlotEigs

    nExp = numel( experiments );

    % Set up figure and axes 
    Fig.nTileX     = 1;
    Fig.nTileY     = 1;
    Fig.units      = 'inches';
    Fig.figWidth   = 5; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .45;
    Fig.deltaY2    = .1;
    Fig.gapX       = .40;
    Fig.gapY       = 0.4;
    Fig.gapT       = 0; 
    Fig.aspectR    = 9 / 16;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 8;
    Fig.tickLength = [ 0.01 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );

    set( gcf, 'currentAxes', ax )

    for iExp = 1 : nExp

        model = demoNLSA_nlsaModel( experiments{ iExp } );
        lambda = getDiffusionEigenvalues( model );

        plot( [ 0 : numel( lambda ) - 1 ], lambda, markers{ iExp } )
    end
         
    grid on
    xlim( [ 0 20 ] )
    ylim( [ 0 1.1 ] )
    legend( labels, 'location', 'northEast' )
    ylabel( '\lambda_j' )
    xlabel( 'j' )

    figFile = fullfile( figDir, 'figLambda.png' );
    print( fig, figFile, '-dpng', '-r300' ) 

end


if ifPlotZ
        
    nExp = numel( experiments );

    % Set up figure and axes 
    Fig.nTileX     = nExp;
    Fig.nTileY     = 1;
    Fig.units      = 'inches';
    Fig.figWidth   = 10; 
    Fig.deltaX     = .65;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = 0.4;
    Fig.gapT       = 0; 
    Fig.aspectR    = 1;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.01 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );


    for iExp = 1 : nExp

        model = demoNLSA_nlsaModel( experiments{ iExp } );
        phi = getDiffusionEigenfunctions( model );
        z = complex( phi( 5000 : 10000, idxPhi{ iExp }( 1 ) ), ...
                     phi( 5000 : 10000, idxPhi{ iExp }( 2 ) ) ) / sqrt( 2 );

        set( gcf, 'currentAxes', ax( iExp ) )
        plot( z, '-' )
        grid on
        xlim( [ -1.5 1.5 ] )
        ylim( [ -1.5 1.5 ] )
        title( labels{ iExp } )
        if iExp == 1
            ylabel( 'Im z' )
        else
            set( gca, 'yTickLabel', [] )
        end
        xlabel( 'Re z' )

    end

    figFile = fullfile( figDir, 'figZ.png' );
    print( fig, figFile, '-dpng', '-r300' ) 

end

if ifPlotAlpha

    nExp = numel( experiments );

    % Set up figure and axes 
    Fig.nTileX     = 1;
    Fig.nTileY     = nExp;
    Fig.units      = 'inches';
    Fig.figWidth   = 6; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = 0.5;
    Fig.gapT       = 0; 
    Fig.aspectR    = 9 / 16;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.01 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );


    for iExp = 1 : nExp

        [ model, In ] = demoNLSA_nlsaModel( experiments{ iExp } );
        [ phi, mu ] = getDiffusionEigenfunctions( model );
        z = complex( phi( :, idxPhi{ iExp }( 1 ) ), ...
                     phi( :, idxPhi{ iExp }( 2 ) ) ) / sqrt( 2 );

        t = ( 0 : nT - 1 ) * In.dt;
        alphaT = zCorr( z, mu, nT );

        set( gcf, 'currentAxes', ax( iExp ) )
        plot( t, real( alphaT ), 'b-' )
        plot( t, imag( alphaT ), 'r-' )
        plot( t, abs( alphaT ), 'k-' )
        
        grid on
        xlim( [ 0 10 ] )
        ylim( [ -1.1 1.1 ] )
        title( labels{ iExp } )
        if iExp == nExp
            xlabel( 't' )
        end
        if iExp == 1
            legend( 'Re \alpha_t', 'Im \alpha_t', '|\alpha_t|', ...
                    'location', 'northeast' )
        end
        set( gca, 'yTick', [ -1 : .2 : 1 ] )
    end
         
    figFile = fullfile( figDir, 'figAlpha.png' );
    print( fig, figFile, '-dpng', '-r300' ) 

end

if ifPlotOmega

    % Set up figure and axes 
    Fig.nTileX     = 1;
    Fig.nTileY     = 1;
    Fig.units      = 'inches';
    Fig.figWidth   = 6; 
    Fig.deltaX     = .5;
    Fig.deltaX2    = .2;
    Fig.deltaY     = .5;
    Fig.deltaY2    = .3;
    Fig.gapX       = .40;
    Fig.gapY       = 0.5;
    Fig.gapT       = 0; 
    Fig.aspectR    = 9 / 16;
    Fig.fontName   = 'helvetica';
    Fig.fontSize   = 12;
    Fig.tickLength = [ 0.01 0 ];
    Fig.visible    = 'on';
    Fig.nextPlot   = 'add'; 

    [ fig, ax ] = tileAxes( Fig );

    set( gcf, 'currentAxes', ax )


    [ model, In ] = demoNLSA_nlsaModel( experiments{ 2 } );
    [ phi, mu ] = getDiffusionEigenfunctions( model );
    z = complex( phi( :, idxPhi{ 2 }( 1 ) ), ...
                 phi( :, idxPhi{ 2 }( 2 ) ) ) / sqrt( 2 );
    
    t = ( 0 : nT - 1 ) * In.dt;

    alphaT = zCorr( z, mu, nT );

    omega = zOmega( z, mu, In.dt );

    eOmegaT = exp( i * omega * t );

    plot( t, real( alphaT ), 'b-' )
    plot( t, real( eOmegaT ), 'r-' )

    legend( 'Re \alpha_t', 'Re e^{i \omega t}', 'location', 'northeast' )
    title( sprintf( '%s;  \\omega = %1.2f', labels{ 2 }, omega ) )
    grid on
    xlim( [ 0 10 ] )
    ylim( [ -1.1 1.1 ] )
    xlabel( 't' )

    figFile = fullfile( figDir, 'figOmega.png' );
    print( fig, figFile, '-dpng', '-r300' ) 

end
    


function a = zCorr( z, mu, nT )

a = zeros( 1, nT );

nS = numel( mu );

zMu = z .* mu;

for iT = 1 : nT
    a( iT ) = zMu( 1 : nS - iT + 1 )' * z( iT : end );
end  

end

function omega = zOmega( z, mu, dt )

w = fdWeights( 4, 'central' ) / dt;
nW   = numel( w );            % number of weights
nS   = numel( z );
nSFD = nS - nW + 1;           % number of output samples

dz = zeros( nSFD, 1 );
for iW = 1 : nW
    dz = dz + w( iW ) * z( iW : iW + nSFD - 1 );
end
zMu = z .* mu;
zMu = zMu( 3 : end - 2 );
omega = sum( imag( zMu ) .*  real( dz ) ) * 2;

end

