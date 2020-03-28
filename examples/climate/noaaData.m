% READ NOAA REANALYSIS DATA AND OUUTPUT IN .MAT FORMAT AS APPROPRIATE FOR NLSA
% CODE
%
% Longitude range is [ 0 359 ] at 1 degree increments
% Latitude range is [ -89 89 ] at 1 degree increments
% Date range is January 18854 to June 2019 at 1 month increements
%
% Modified 2020/03/27

%% FILE AND VARIABLE NAMES
dataDirIn  = '/Volumes/TooMuch/physics/climate/data/noaa'; % input data dir.
filenameIn = 'sst.mnmean.v4-4.nc';    % filename base for input data
fldIn      = 'sst';                   % field name in NetCDF file 
experiment = 'noaa';                  % label for data analysis experiment 
fld        = 'sst';                   % output label for field 

%% REGIONS
% Indo-Pacific domain
%xLim = [ 28 290 ]; % longitude limits
%yLim = [ -60 20 ]; % latitude limits

% Nino 3.4 region
xLim = [ 190 240 ]; % longitude limits 
yLim = [ -5 5 ];    % latitude limits

% Nino 1+2 region
%xLim = [ 270 280 ]; % longitude limits 
%yLim = [ -10 0 ];   % latitude limits

% Nino 3 region
%xLim = [ 210 270 ]; % longitude limits 
%yLim = [ -5 5 ];    % latitude limits

% Nino 4 region
%xLim = [ 160 210 ]; % longitude limits 
%yLim = [ -5 5 ];    % latitude limits

%% TIME INTERVALS
% ENSO lifecycle
tLim = { '187001' '201906' }; % training time limits

% Hindcast
%tLim = { '187001' '200712' }; % training time limits 
%tLim = { '200801' '201906' }; % verification time limits 

% "Operational" forecast
%tLim = { '187001' '201906' }; % training time limits 
%tLim = { '201705' '201906' }; % initialization time limits

tClim   = { '198101' '201012' }; % time limits for climatology removal
tStart  = '185401';              % start time in nc file 
tFormat = 'yyyymm';              % time format

%% SCRIPT EXECUTION OPTIONS
ifCenter      = false;  % remove global climatology
ifWeight      = true;   % perform area weighting
ifCenterMonth = true;   % remove monthly climatology 
ifAverage     = true;   % perform area averaging
ifNormalize   = false;  % normalize to unit L2 norm


%% READ DATA
% Check for consistency of climatological averaging
if ifCenter & ifCenterMonth
    error( [ 'Global and monthly climatology removal cannot be ' ...
             'simultaneously selected' ] )
end

% Append 'a' to field string if outputting anomalies
if ifCenter
    fldStr = [ fld 'a' ];
else
    fldStr = fld;
end

% Append 'ma' to field string if outputting monthly anomalies
if ifCenterMonth
    fldStr = [ fld 'ma' ];
else
    fldStr = fld;
end

% Append 'w' if performing area weighting
if ifWeight
    fldStr = [ fldStr 'w' ];
end

% Append 'av' if outputting area average
if ifAverage
    fldStr = [ fldStr 'av' ];
end

% Append 'n' if normalizing
if ifNormalize
    fldStr = [ fldStr 'n' ];
end

% Append time limits for climatology 
if ifCenter | ifCenterMonth 
    fldStr = [ fldStr '_' tClim{ 1 } '-' tClim{ 2 } ];
end 

% Output directory
dataDir = fullfile( pwd, ...
                    'data/raw', ...
                    experiment, ...
                    fldStr, ...
                    [ sprintf( 'x%i-%i', xLim ) ...
                      sprintf( '_y%i-%i', yLim ) ...
                      '_' tLim{ 1 } '-' tLim{ 2 } ] );
if ~isdir( dataDir )
    mkdir( dataDir )
end

% Number of samples and starting time index
% Indices are based at 0 to conform with NetCDF library
limNum = datenum( tLim, tFormat );
climNum = datenum( tClim, tFormat );
startNum = datenum( tStart, tFormat );
nT    = months( limNum( 1 ), limNum( 2 ) ) + 1;
nTClim = months( climNum( 1 ), climNum( 2 ) ) + 1;
idxT0 = months( startNum, limNum( 1 ) );  
idxTClim0 = months( startNum, climNum( 1 ) ); 

% Open netCDF file, find variable IDs
ncId   = netcdf.open( fullfile( dataDirIn, filenameIn ) );
idLon  = netcdf.inqVarID( ncId, 'lon' );
idLat  = netcdf.inqVarID( ncId, 'lat' );
idFld  = netcdf.inqVarID( ncId, fld );

% Create region mask
lon = netcdf.getVar( ncId, idLon );
lat = netcdf.getVar( ncId, idLat );
nX  = length( lon );
nY  = length( lat );
[ X, Y ] = ndgrid( lon, lat );
rng = netcdf.getAtt( ncId, idFld, 'valid_range' );
fldRef = netcdf.getVar( ncId, idFld, [ 0 0 0 ], [ nX nY 1 ] ); 
ifXY = X >= xLim( 1 ) & X <= xLim( 2 ) ...
     & Y >= yLim( 1 ) & Y <= yLim( 2 ) ...
     & fldRef >= rng( 1 ) & fldRef <= rng( 2 );
iXY = find( ifXY( : ) );
nXY = length( iXY );

%  Retrieve data
fld = netcdf.getVar( ncId, idFld, [ 0 0 idxT0 ], [ nX nY nT ] );
fld = reshape( fld, [ nX * nY nT ] );
fld = fld( iXY, : );

% If requested, weigh the data by the (normalized) grid cell surface areas. 
% Surface area calculation is approximate as it treats Earth as spherical
if ifWeight

    % Convert to radians and augment grid periodically 
    dLon = [ -2; lon; 360 ] * pi / 180; 
    dLat = [ 90; lat; -90 ] * pi / 180;

    % Compute grid coordinate differences
    dLon = ( dLon( 1 : end - 1 ) + dLon( 2 : end ) ) / 2;
    dLon = dLon( 2 : end ) - dLon( 1 : end - 1 );
    dLat = ( dLat( 1 : end - 1 ) + dLat( 2 :end ) ) / 2;
    dLat = dLat( 2 : end ) - dLat( 1 : end - 1 );
    dLat = abs( dLat ) .* cos( dLat );

    % Compute surface area weights
    w = bsxfun( @times, dLon, dLat' );
    w = w( ifXY );
    w = sqrt( w / sum( w ) * nXY );
      
    % Weigh the data
    fld = bsxfun( @times, fld, w );
end

% If requested, subtract climatology
if ifCenter
    cli = netcdf.getVar( ncId, idFld, [ 0 0 idxTClim0 ], [ nX nY nTClim ] );
    cli = reshape( cli, [ nX * nY nTClim ] );
    cli = cli( iXY, : );
    if ifWeight
        cli = bsxfun( @times, cli, w );
    end
    cli = mean( cli, 2 );
    fld = bsxfun( @minus, fld, cli );
end

% If requested, subtract mnthly climatology
if ifCenterMonth
    cliData = netcdf.getVar( ncId, idFld, [ 0 0 idxTClim0 ], [ nX nY nTClim ] );
    cliData = reshape( cliData, [ nX * nY nTClim ] );
    cliData = cliData( iXY, : );
    if ifWeight
        cliData = bsxfun( @times, cliData, w );
    end
    cli = zeros( nXY, 12 );
    for iM = 1 : 12
        cli( :, iM ) = mean( cliData( :, iM : 12 : end ), 2 );
    end
    idxM0 = month( limNum( 1 ) ); 
    for iM = 1 : 12
        idxM = mod( idxM0 + iM - 2, 12 ) + 1; 
        fld( :, iM : 12 : end ) = bsxfun( @minus, fld( :,  iM : 12 : end ), ...
                                                  cli( :, idxM ) ); 
    end  
end

% NetCDF file no longer needed
netcdf.close( ncId );

% If requested, perform area averaging
if ifAverage
    fld = mean( fld, 1 );
    if ifCenter || ifCenterMonth
        cli = mean( cli, 1 );
    end
end

% Output data dimension
nD = size( fld, 1 );

% If requested, normalize by RMS climatology norm
if ifNormalize
    l2Norm = norm( cli( : ), 2 ) / sqrt( nTClim );
    fld = fld / l2Norm;
end

% Save grid coordinates and area mask
gridFile = fullfile( dataDir, 'dataGrid.mat' );
save( gridFile, 'lat', 'lon', 'ifXY', 'fldStr', 'nD', '-v7.3' )  

% Save output data and attributes
fldFile = fullfile( dataDir, 'dataX.mat' );
x = double( fld ); % for compatibility with NLSA code
varList = { 'x' 'idxT0' };
if ifCenter || ifCenterMonth
    varList = [ varList 'cli' 'idxTClim0' 'nTClim' ];
end
if ifWeight
    varList = [ varList 'w' ];
end
if ifNormalize
    varList = [ varList 'l2Norm' ];
end
save( fldFile, varList{ : },  '-v7.3' )  

