%% READ NOAA REANALYSIS DATA AND OUUTPUT IN .MAT FORMAT APPROPRIATE FOR NLSA
%% CODE
%
% Modified 2019/06/20

dataDirIn  = '/Volumes/TooMuch/physics/climate/data/noaa'; % directory name for input data
filenameIn = 'sst.mnmean.v3.nc';    % filename base for input data
fldIn      = 'sst';                 
experiment = 'noaa';                % label for data analysis experiment 
fld        = 'sst';                 % label for field 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Indo-Pacific domain
xLim       = [ 28 290 ];            % longitude limits
yLim       = [ -60  20  ];          % latitude limits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nino 3.4 region
%xLim = [ 190 240 ];                 % longitude limits 
%yLim = [ -5 5 ];                    % latitude limits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tLim       = { '187001' '198712' }; % training time limits 
tLim       = { '198712' '201902' }; % verification time limits 
tClim      = { '198101' '201012' }; % time limits for climatology
tStart     = '185401';              % start time in nc file 
tFormat    = 'yyyymm';              % time format

ifCenter   = false;                 % remove climatology
ifWeight   = true;                  % perform area weighting
ifAverage  = false;                 % perform area averaging

% Append 'a' to field string if outputting anomalies
if ifCenter
    fldStr = [ fld 'a' ];
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

% Append time limits for climatology if different from output time limits
if ifCenter && any( ~strcmp( tLim, tClim ) )
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
ncFile = fullfile( dataDirIn, filenameIn );
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

% If requested, subtract climatology
if ifCenter
    cli = netcdf.getVar( ncId, idFld, [ 0 0 idxTClim0 ], [ nX nY nTClim ] );
    cli = reshape( cli, [ nX * nY nTClim ] );
    cli = cli( iXY, : );
    cli = mean( cli, 2 );
    fld = bsxfun( @minus, fld, cli );
end

% NetCDF file no longer needed
netcdf.close( ncId );

% If requested, weigh the data by the (normalized) grid cell surface areas. 
% Surface area calculation is approximate as it treats Earth as spherical
if ifWeight

    % Convert to ratians and augment grid periodically 
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
    w = w / sum( w );
      
    % Weigh the data
    fld = bsxfun( @times, fld, w );
end

% If requested, perform area averaging
if ifAverage
    fld = mean( fld, 1 );
    if ifCenter
        cli = mean( cli, 1 );
    end
end

% Output data dimension
nD = size( fld, 1 );

% Save grid coordinates and area mask
gridFile = fullfile( dataDir, 'dataGrid.mat' );
save( gridFile, 'lat', 'lon', 'ifXY', 'fldStr', 'nD', '-v7.3' )  

% Save output data and attributes
fldFile = fullfile( dataDir, 'dataX.mat' );
x = fld; % for compatibility with NLSA code
varList = { 'x' 'idxT0' };
if ifCenter
    varList = [ varList 'cli' 'idxTClim0' 'nTClim' ];
end
if ifWeight
    varList = [ varList 'w' ];
end
save( fldFile, varList{ : },  '-v7.3' )  

