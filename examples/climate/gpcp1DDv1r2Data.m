%% READ GPCP 1 DEGREE DAILY (1DD) DATA AND OUUTPUT IN .MAT FORMAT 
%% APPROPRIATE FOR NLSA CODE
%
% Longitude range is [ 0.5 359.5 ] at 1 degree increments
% Latitude range is [ -89.5 89.5 ] at 1 degree increments
%
% Modified 2019/07/30

dataDirIn  = '/Volumes/TooMuch/physics/climate/data/gpcp/1dd_v1.2/'; % directory name for input data
filenameIn = 'GPCP_1DD_v1.2_199610-201510.nc';    % filename base for input data
fldIn      = 'PREC';                 
experiment = 'gpcp_1dd_v1.2';                % label for data analysis experiment 
fld        = 'precip';                 % label for field 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% South Asian summer Monsoon domain
xLim       = [  30 160 ];          % longitude limits
yLim       = [ -20  40 ];          % latitude limits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tLim       = { '19961001' '20101231' }; % training time limits 
tLim      = { '20110101' '20151031' }; % verification time limits 
tClim      = { '19961001' '20101231' }; % time limits for climatology
tStart     = '19961001';              % start time in nc file 
tFormat    = 'yyyymmdd';              % time format

ifCenter      = false;                 % remove climatology
ifWeight      = false;                 % perform area weighting
ifAverage     = false;                % perform area averaging
ifNormalize   = false;                 % normalize to unit L2 norm

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

% Append 'n' if normalizing
if ifNormalize
    fldStr = [ fldStr 'n' ];
end

% Append time limits for climatology 
if ifCenter 
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
nT    = daysact( limNum( 1 ), limNum( 2 ) ) + 1;
nTClim = daysact( climNum( 1 ), climNum( 2 ) ) + 1;
idxT0 = daysact( startNum, limNum( 1 ) );  
idxTClim0 = daysact( startNum, climNum( 1 ) ); 

% Open netCDF file, find variable IDs
ncId   = netcdf.open( fullfile( dataDirIn, filenameIn ) );
idLon  = netcdf.inqVarID( ncId, 'lon' );
idLat  = netcdf.inqVarID( ncId, 'lat' );
idFld  = netcdf.inqVarID( ncId, fldIn );

% Retrieve grid 
lon = netcdf.getVar( ncId, idLon );
lat = netcdf.getVar( ncId, idLat );
[ lat, idxSrt ] = sort( lat, 'ascend' );
nX  = length( lon );
nY  = length( lat );

% Create region mask 
% We only retain grid points with physical values for the entire temporal
% extent of the dataset. 
fld = netcdf.getVar( ncId, idFld );
fld = fld( :, idxSrt, : );
[ X, Y ] = ndgrid( lon, lat );
ifXY = X >= xLim( 1 ) & X <= xLim( 2 ) ...
     & Y >= yLim( 1 ) & Y <= yLim( 2 ) ...
     & all( fld > -1000, 3 );  
iXY = find( ifXY( : ) );
nXY = length( iXY ); 

% Extract data in region mask 
fld = netcdf.getVar( ncId, idFld, [ 0 0 idxT0 ], [ nX nY nT ] );
fld = fld( :, idxSrt, : );
fld = reshape( fld, [ nX * nY nT ] );
fld = fld( iXY, : );

% If requested, weigh the data by the (normalized) grid cell surface areas. 
% Surface area calculation is approximate as it treats Earth as spherical
if ifWeight

    % Convert to radians and augment grid periodically 
    dLon = [ -0.5; lon; 360.5 ] * pi / 180; 
    dLat = [ 90.5; lat; -90.5 ] * pi / 180;

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
    cli = cli( idxSrt, : );
    cli = reshape( cli, [ nX * nY nTClim ] );
    cli = cli( iXY, : );
    if ifWeight
        cli = bsxfun( @times, cli, w );
    end
    cli = mean( cli, 2 );
    fld = bsxfun( @minus, fld, cli );
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

% If requested, normalize by RMS climatology norm
if ifNormalize
    l2Norm = norm( cli( : ), 2 ) / sqrt( nTClim );
    fld = fld / l2Norm;
end


% NetCDF file no longer needed
netcdf.close( ncId );

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
if ifNormalize
    varList = [ varList 'l2Norm' ];
end
save( fldFile, varList{ : },  '-v7.3' )  

