function Data = noaaData( DataSpecs )
% NOAADATA Read NOAA 20th Century Reanalysis data and output in format 
% appropriate for NLSA code.
% 
% DataSpecs is a data structure containing the specifications of the data to
% be read. It has the following fields:
%
% In.dir:             Input directory name
% In.file:            Input filename
% In.var:             Variable to be read
% Out.dir:            Output directory name
% Out.fld:            Output label 
% Time.tFormat:       Format of serial date numbers (e.g, 'yyyymm')
% Time.tLim:          Cell array of strings with time limits 
% Time.tClim:         Cell array of strings with time limits for climatology 
% Domain.xLim:        Longitude limits
% Domain.yLim:        Latitude limits
% Opts.ifCenter:      Remove global climatology if true 
% Opts.ifWeight:      Perform area weighting if true 
% Opts.ifCenterMonth: Remove monthly climatology if true 
% Opts.ifNormalize:   Standardize data to unit L2 norm if true
% Opts.ifWrite:       Write data to disk
%
% Longitude range is [ 0 359 ] at 1 degree increments
% Latitude range is [ -89 89 ] at 1 degree increments
% Date range is January 18854 to June 2019 at 1 month increements
%
% Outpur argument Data contains the data read and associated attributes.
%
% Modified 2020/03/28

%% UNPACK INPUT DATA STRUCTURE FOR CONVENIENCE
In     = DataSpecs.In;
Out    = DataSpecs.Out; 
Time   = DataSpecs.Time;
Domain = DataSpecs.Domain;
Opts   = DataSpecs.Opts;

%% READ DATA
% Check for consistency of climatological averaging
if Opts.ifCenter && Opts.ifCenterMonth
    error( [ 'Global and monthly climatology removal cannot be ' ...
             'simultaneously selected' ] )
end

% Append 'a' to field string if outputting anomalies
if Opts.ifCenter
    fldStr = [ Out.fld 'a' ];
else
    fldStr = Out.fld;
end

% Append 'ma' to field string if outputting monthly anomalies
if Opts.ifCenterMonth
    fldStr = [ Out.fld 'ma' ];
else
    fldStr = Out.fld;
end

% Append 'w' if performing area weighting
if Opts.ifWeight
    fldStr = [ fldStr 'w' ];
end

% Append 'av' if outputting area average
if Opts.ifAverage
    fldStr = [ fldStr 'av' ];
end

% Append 'n' if normalizing
if Opts.ifNormalize
    fldStr = [ fldStr 'n' ];
end

% Append time limits for climatology 
if Opts.ifCenter || Opts.ifCenterMonth 
    fldStr = [ fldStr '_' Time.tClim{ 1 } '-' Time.tClim{ 2 } ];
end 

% Output directory
dataDir = fullfile( Out.dir, ...
                    fldStr, ...
                    [ sprintf( 'x%i-%i',  Domain.xLim ) ...
                      sprintf( '_y%i-%i', Domain.yLim ) ...
                      '_' Time.tLim{ 1 } '-' Time.tLim{ 2 } ] );
if Opts.ifWrite && ~isdir( dataDir )
    mkdir( dataDir )
end

% Number of samples and starting time index
% Indices are based at 0 to conform with NetCDF library
limNum = datenum( Time.tLim, Time.tFormat );
climNum = datenum( Time.tClim, Time.tFormat );
startNum = datenum( Time.tStart, Time.tFormat );
nT    = months( limNum( 1 ), limNum( 2 ) ) + 1;
nTClim = months( climNum( 1 ), climNum( 2 ) ) + 1;
idxT0 = months( startNum, limNum( 1 ) );  
idxTClim0 = months( startNum, climNum( 1 ) ); 

% Open netCDF file, find variable IDs
ncId   = netcdf.open( fullfile( In.dir, In.file ) );
idLon  = netcdf.inqVarID( ncId, 'lon' );
idLat  = netcdf.inqVarID( ncId, 'lat' );
idFld  = netcdf.inqVarID( ncId, In.var );

% Create region mask
lon = netcdf.getVar( ncId, idLon );
lat = netcdf.getVar( ncId, idLat );
nX  = length( lon );
nY  = length( lat );
[ X, Y ] = ndgrid( lon, lat );
rng = netcdf.getAtt( ncId, idFld, 'valid_range' );
fldRef = netcdf.getVar( ncId, idFld, [ 0 0 0 ], [ nX nY 1 ] ); 
ifXY = X >= Domain.xLim( 1 ) & X <= Domain.xLim( 2 ) ...
     & Y >= Domain.yLim( 1 ) & Y <= Domain.yLim( 2 ) ...
     & fldRef >= rng( 1 ) & fldRef <= rng( 2 );
iXY = find( ifXY( : ) );
nXY = length( iXY );

%  Retrieve data
fld = netcdf.getVar( ncId, idFld, [ 0 0 idxT0 ], [ nX nY nT ] );
fld = reshape( fld, [ nX * nY nT ] );
fld = fld( iXY, : );

% If requested, weigh the data by the (normalized) grid cell surface areas. 
% Surface area calculation is approximate as it treats Earth as spherical
if Opts.ifWeight

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
    w = dLon .* dLat';
    w = w( ifXY );
    w = sqrt( w / sum( w ) * nXY );
      
    % Weigh the data
    fld = fld .* w;
end

% If requested, subtract climatology
if Opts.ifCenter
    cli = netcdf.getVar( ncId, idFld, [ 0 0 idxTClim0 ], [ nX nY nTClim ] );
    cli = reshape( cli, [ nX * nY nTClim ] );
    cli = cli( iXY, : );
    if Opts.ifWeight
        cli = cli .* w;
    end
    cli = mean( cli, 2 );
    fld = fld - cli;
end

% If requested, subtract mnthly climatology
if Opts.ifCenterMonth
    cliData = netcdf.getVar( ncId, idFld, [ 0 0 idxTClim0 ], [ nX nY nTClim ] );
    cliData = reshape( cliData, [ nX * nY nTClim ] );
    cliData = cliData( iXY, : );
    if Opts.ifWeight
        cliData = cliData .* w;
    end
    cli = zeros( nXY, 12 );
    for iM = 1 : 12
        cli( :, iM ) = mean( cliData( :, iM : 12 : end ), 2 );
    end
    idxM0 = month( limNum( 1 ) ); 
    for iM = 1 : 12
        idxM = mod( idxM0 + iM - 2, 12 ) + 1; 
        fld( :, iM : 12 : end ) = fld( :,  iM : 12 : end ) - cli( :, idxM ); 
    end  
end

% NetCDF file no longer needed
netcdf.close( ncId );

% If requested, perform area averaging
if Opts.ifAverage
    fld = mean( fld, 1 );
    if Opts.ifCenter || Opts.ifCenterMonth
        cli = mean( cli, 1 );
    end
end

% Output data dimension
nD = size( fld, 1 );

% If requested, normalize by RMS climatology norm
if Opts.ifNormalize
    l2Norm = norm( cli( : ), 2 ) / sqrt( nTClim );
    fld = fld / l2Norm;
end

%% RETURN AND WRITE DATA
% Coordinates and area mask
gridVarList = { 'lat', 'lon', 'ifXY', 'fldStr', 'nD' };
if Opts.ifWrite
    gridFile = fullfile( dataDir, 'dataGrid.mat' );
    save( gridFile, gridVarList{ : }, '-v7.3' )  
end

% Output data and attributes
x = double( fld ); % for compatibility with NLSA code
varList = { 'x' 'idxT0' };
if Opts.ifCenter || Opts.ifCenterMonth
    varList = [ varList 'cli' 'idxTClim0' 'nTClim' ];
end
if Opts.ifWeight
    varList = [ varList 'w' ];
end
if Opts.ifNormalize
    varList = [ varList 'l2Norm' ];
end

if Opts.ifWrite
    fldFile = fullfile( dataDir, 'dataX.mat' );
    save( fldFile, varList{ : },  '-v7.3' )  
end

% If needed, assemble data and attributes into data structure and return
if nargout > 0
    varList = [ varList gridVarList ];
    nVar = numel( varList );
    vars = cell( 1, nVar );
    for iVar = 1 : nVar
       vars{ iVar } = eval( varList{ iVar } );
    end
    Data = cell2struct( vars, varList, 2 );
end
