dataDirIn  = fullfile( '../../../', 'data' );
fBase      = 'prec_imd_onedeg_';
fldName    = 'prec';             
xName      = 'lat';
yName      = 'lon';
tName      = 'time';
xLim       = [];                  % longitude limitsi (empty if full domain)
yLim       = [];                  % latitude limits 
yrLim      = [ 2005 2012 ];       % time limits (years) 
dateLim    = { '0501' '1001' };
idxT       = 1;                   % time index in nc file
idxX       = 3;                   % longitude index in nc file
idxY       = 2;                   % latitude  index in nc file
idxFld     = 4;                   % field index in nc file
ifAnom     = false;               % subtract climatology

lbl = fldName; 
if ifAnom
    lbl = [ lbl '_anom' ];
end

dateLimN = datenum( dateLim, 'mmdd' )

% Determine total number of gridpoints and samples
nYr = yrLim( 2 ) - yrLim( 1 ) + 1;
for iYr = 1 : nYr
    yr = yrLim( 1 ) + iYr - 1;
    dateStr{ 1 } = [ dateLim{ 1 } int2str( yr ) ];
    dateStr{ 2 } = [ dateLim{ 2 } int2str( yr ) ];
    xyr = strjoin( dateStr, '-' );
    if ~isempty( xLim )
        xyr = [ xyr ...
                sprintf( '_x%i-%i_y%i-%i', xLim( 1 ), ...
                                           xLim( 2 ), ...
                                           yLim( 1 ), ...
                                           yLim( 2 ) ) ];
    end
    dataDir = fullfile( 'data/raw', lbl, xyr );
    mkdir( dataDir )


    fName = fullfile( dataDirIn, sprintf( '%s%i.nc', fBase, yr ) );
    disp( fName )
    I = ncinfo( fName ); 
    nSY = I.Dimensions( idxT ).Length;
    disp( sprintf( 'Number of samples in year = %i', nSY ) )
    if iYr == 1
        % Get area mask
        nX = I.Dimensions( idxX ).Length;
        nY = I.Dimensions( idxY ).Length;
        Grid.x = ncread( fName, xName );
        Grid.y = ncread( fName, yName );
        f = ncread( fName, fldName );
        f = squeeze( f( :, :, 1 ) );
        ifXY = ~isnan( f );
        [ X, Y ] = ndgrid( Grid.x, Grid.y );
        if ~isempty( xLim )
            ifXY = ifXY ...
                 & X >= xLim( 1 ) & X <= xLim( 2 ) ...
                 & Y >= yLim( 1 ) & Y <= yLim( 2 );
        end
        nD = nnz( ifXY );    
        disp( sprintf( 'Number of gridpoints = %i', nD ) )
    elseif iYr == nYr
        gridFile    = [ dataDir, '/dataGrid.mat' ];
        save( gridFile, 'X', 'Y', 'ifXY', 'nD' )  
    end
    t = ncread( fName, tName );
    iS1 = find(   day( t ) == day( dateLimN( 1 ) ) ...
                & month( t ) == month( dateLimN( 1 ) ), 1, 'first' );
    iS2 = find(   day( t ) == day( dateLimN( 2 ) ) ...
                & month( t ) == month( dateLimN( 2 ) ), 1, 'first' );
    nS = iS2 - iS1 + 1;
    disp( sprintf( 'Number of samples in interval = %i', nS ) )
    t = t( iS1 : iS2 );
    x = zeros( nD, nS );
    fYr = ncread( fName, fldName );
    fYr = fYr( :, :, iS1 : iS2 );
    for iS = 1 : nS
        f = squeeze( fYr( :, :, iS ) );
        x( :, iS ) = f( ifXY );
        if any( isnan( x( :, iS ) ) )
            error( 'Incompatile area masks' )
        end
    end

    % Compute climatology
    mu = mean( x, 2 );
    if ifAnom
        x = bsxfun( @minus, x, mu );
    end
    xNorm = sum( x.^ 2, 1 );
    xAv   = sqrt( sum( xNorm ) / nS );
    xNorm = sqrt( xNorm ); 

    anomFile    = [ dataDir, '/dataX.mat' ];
    save( anomFile, 'x', 't', 'nD', 'nS', 'mu', 'xNorm', 'xAv' )  
end 

