%rootDir    = '/Volumes/TooMuch/physics/climate/monsoon.1'
%dataDirIn  = fullfile( rootDir, 'data' );
dataDirIn  = pwd;
fBase      = 'prec_imd_onedeg_';
fldName    = 'prec';             
xName      = 'lat';
yName      = 'lon';
tName      = 'time';
xLim       = [];                  % longitude limitsi (empty if full domain)
yLim       = [];                  % latitude limits 
yrLim      = [ 2012 2013 ];       % time limits (years) 
idxT       = 1;                   % time index in nc file
idxX       = 3;                   % longitude index in nc file
idxY       = 2;                   % latitude  index in nc file
idxFld     = 4;                   % field index in nc file
ifAnom     = false;               % subtract climatology

lbl = fldName; 
if ifAnom
    lbl = [ lbl '_anom' ];
end
xyr = sprintf( 'yr%i-%i', yrLim( 1 ), yrLim( 2 ) );
if ~isempty( xLim )
    xyr = [ xyr ...
            sprintf( '_x%i-%i_y%i-%i', xLim( 1 ), ...
                                       xLim( 2 ), ...
                                       yLim( 1 ), ...
                                       yLim( 2 ) ) ];
end
dataDir = fullfile( 'data/raw', lbl, xyr );
mkdir( dataDir )

% Determine total number of gridpoints and samples
nYr = yrLim( 2 ) - yrLim( 1 ) + 1;
nS = 0; 
for iYr = 1 : nYr
    fName = sprintf( '%s%i.nc', fBase, yrLim( 1 ) + iYr - 1 );
    I = ncinfo( fName ); 
    if iYr == 1
        nX = I.Dimensions( idxX ).Length;
        nY = I.Dimensions( idxY ).Length;
    end
    nS = nS + I.Dimensions( idxT ).Length;
end
disp( sprintf( 'Number of samples = %i', nS ) )

% Get area mask
fName = sprintf( '%s%i.nc', fBase, yrLim( 1 ) );
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

% Read data
x = zeros( nD, nS );
t = zeros( 1, nS );
iS = 1;
for iYr = 1 : nYr
    fName = sprintf( '%s%i.nc', fBase, yrLim( 1 ) + iYr - 1 );
    disp( [ 'Reading input file ' fName ] )
    fYr = ncread( fName, fldName );
    nSYr = size( fYr, 3 );
    t( iS : iS + nSYr - 1 ) = ncread( fName, tName );
    for iSYr = 1 : nSYr;
        f = squeeze( fYr( :, :, iSYr ) );
        x( :, iS ) = f( ifXY );
        if any( isnan( x( :, iS ) ) )
            error( 'Incompatile area masks' )
        end
        iS = iS + 1;
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

gridFile    = [ dataDir, '/dataGrid.mat' ];
save( gridFile, 'X', 'Y', 'ifXY', 'nD' )  
   
