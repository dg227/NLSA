% This script reads NetCDF data from CCSM4 control run b40.1850 and outputs it
% in .mat format readable by the NLSA code.
%
% WARNING: In this version, the start and end years must match with start
% and end years in the .nc files.
%
% Modified 2016/06/03

dataDirIn  = '/Volumes/TooMuch/physics/climate/data/ccsm4'; % directory name for input data
fileBase   = 'b40.1850.track1.1deg.006.pop.h.SST.'; % filename base for input data
experiment = 'b40.1850';            % label for data analysis experiment 
fld        = 'SST';                 % label for field 
xLim       = [ 28 290 ];           % longitude limits
yLim       = [ -60  20  ];           % latitude limits
yrLim      = [ 1   1300 ];          % time limits (years) 
yrBatch    = 100;                   % (nominal) years per input file
idxX       = 20;                    % longitude index in nc file
idxY       = 19;                    % latitude  index in nc file
idxT       = 54;                    % time index in nc file
idxA       = 17;                    % area index in nc file
idxM       = 15;                    % region mask index in nc file
idxLbl     = 18;                    % field index in nc file
ifCenter   = true;                 % remove climatology

if ifCenter
    fldStr = strcat( fld, 'A' );
else
    fldStr = fld;
end

dataDir = fullfile( './data/raw', ...
                    experiment, ...
                    fldStr, ...
                    [ sprintf( 'x%i-%i', xLim ), ...
                      sprintf( '_y%i-%i', yLim ), ...
                      sprintf( '_yr%i-%i', yrLim ) ] );
if ~isdir( dataDir )
    mkdir( dataDir )
end

yrStart = yrLim( 1 );
iTStart = 1;
nT      = ( yrLim( 2 ) - yrLim( 1 ) + 1 ) * 12; % number of months
 
while yrStart < yrLim( 2 )

    if yrStart == 1
        yrEnd   = yrStart + ( yrBatch - 1 ) - 1;
        iTEnd   = iTStart + ( yrBatch - 1 ) * 12 - 1;
        nTBatch = ( yrBatch - 1 ) * 12;      % number of months per batch
    elseif yrStart == 1200 
        yrEnd = yrStart + ( yrBatch + 1 ) - 1;
        iTEnd = iTStart + ( yrBatch + 1 ) * 12 - 1;
        nTBatch = ( yrBatch + 1 ) * 12;      % number of months per batch
    else
        yrEnd = yrStart + yrBatch - 1;
        iTEnd = iTStart + yrBatch * 12 - 1;
        nTBatch = yrBatch * 12;      % number of months per batch
    end
    ncFile = fullfile( dataDirIn,  ...
                       experiment, ... 
                       [ fileBase, ...
                         sprintf( '%04i01-%04i12.nc', yrStart, yrEnd ) ] );

    
    if iTStart == 1
        tic
        disp( sprintf( 'Extracting gridpoints from file %s', ncFile ) )
 
        x    = ncread( ncFile, 'TLONG' ); % longitudes
        y    = ncread( ncFile, 'TLAT' );  % latitudes
        w    = ncread( ncFile, 'TAREA' ); % grid cell areas
        ifXY = ncread( ncFile, 'REGION_MASK' ) ~= 0; % nonzero values are ocean gridpoints

        ifXY = x >= xLim( 1 ) & x <= xLim( 2 ) ...
             & y >= yLim( 1 ) & y <= yLim( 2 ) ...
             & ifXY;             
        nXY = numel( ifXY );   % number of gridpoints 
        iXY = find( ifXY( : ) ); % extract linear indices from area mask
        nD       = nnz( ifXY ); % data dimension  
           
        w  = w( ifXY );
        w  = sqrt( w / sum( w ) );

        disp( sprintf( '%i unmasked gridpoints (data space dimension)', nD ) )
        disp( sprintf( '%1.3g max area weight', max( w ) ) )
        disp( sprintf( '%1.3g min area weight', min( w ) ) )
    
        gridFile    = [ dataDir, '/dataGrid.mat' ];
        save( gridFile, 'x', 'y', 'ifXY', 'w', 'fldStr', 'nD', '-v7.3' )  
        toc   

        % Data arrays (note, x is no longer longitude)
        mu = zeros( nD, 1 );
        x  = zeros( nD, nT );
        t  = zeros( 1, nT );
        yrStart = yrEnd + 1;
 
    end
    
    disp( sprintf( 'Extracting data from file %s', ncFile ) )
    tic

    t( iTStart : iTEnd ) = ncread( ncFile, 'time' )';
  
    xBatch = ncread( ncFile, fld );
    xBatch = reshape( xBatch, [ nXY nTBatch ] );
    xBatch = bsxfun( @times, xBatch( iXY, : ), w );
    mu = mu + sum( xBatch, 2 );
    x( :, iTStart : iTEnd ) = xBatch;

    iTStart = iTEnd + 1;
    yrStart = yrEnd + 1;
    
    toc
end

disp( 'Computing climatology...' )
mu = mu / nT;

if ifCenter
    disp( 'Centering data...' )
    x = bsxfun( @minus, x, mu );
end

disp( 'Computing norms...' )
l2NormT = sum( x .^ 2, 1 );
l2Norm  = sqrt( sum( l2NormT ) / nT );
l2NormT = sqrt( l2NormT ); 

disp( 'Writing data...' )
tic
anomFile    = [ dataDir, '/dataX.mat' ];
save( anomFile, 'x', 't', 'nD', 'nT', 'l2NormT', 'l2Norm', 'mu', '-v7.3' )  
toc
