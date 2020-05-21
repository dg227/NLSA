function Data = importData_ersst( DataSpecs )
% IMPORTDATA_ERSST Read monthly data from ERSST reanalysis netCDF files, and 
% output in format appropriate for NLSA code.
% 
% DataSpecs is a data structure containing the specifications of the data to
% be read. 
%
% Data is a data structure containing the data read and associated attributes.
%
% DataSpecs has the following fields:
%
% In.dir:             Input directory name
% In.file:            Input filename base
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
% Opts.ifDetrend:     Remove linear trend if true
% Opts.ifNormalize:   Standardize data to unit L2 norm if true
% Opts.ifWrite:       Write data to disk
% Opts.ifOutputData:  Only data attributes are returned if set to false
%
% If the requested date range preceeds/exceeds the available limits, a
% warning message is displayed and the additional samples are set to 0. 
% 
% Longitude range is [ 0 358 ] 
% Latitude range is [ -88 88 ] 
%
% Modified 2020/05/20



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
ifClim = Opts.ifCenter || Opts.ifCenterMonth;

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

% Append 'l' to field string if linearly detrending the data
if Opts.ifDetrend
    fldStr = [ fldStr 'l' ];
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
if ifClim 
    fldStr = [ fldStr '_' Time.tClim{ 1 } '-' Time.tClim{ 2 } ];
end 

% Check if region mask is to be read
ifMsk = isfield( In, 'msk' );

% Check if area weights are to be read
ifArea = isfield( In, 'area' );

% Output directory
dataDir = fullfile( Out.dir, ...
                    fldStr, ...
                    [ sprintf( 'x%i-%i',  Domain.xLim ) ...
                      sprintf( '_y%i-%i', Domain.yLim ) ...
                      '_' Time.tLim{ 1 } '-' Time.tLim{ 2 } ] );
if Opts.ifWrite && ~isdir( dataDir )
    mkdir( dataDir )
end


% Determine available files and number of monthly samples in each file
files = dir( fullfile( In.dir, [ In.file '*.nc' ] ) );
nFile = numel( files );
nTFiles = zeros( 1, nFile );
for iFile = 1 : nFile

    % Open netCDF file, find number of samples
    ncId   = netcdf.open( fullfile( In.dir, files( iFile ).name ) );
    idTime = netcdf.inqDimID( ncId, 'time' );
    [ ~, nTFiles( iFile ) ] = netcdf.inqDim( ncId, idTime );
   
    % Close currently open file
    netcdf.close( ncId );
end

% Create partition representing how samples are distributed among files 
partitionT = nlsaPartition( 'idx', cumsum( nTFiles ) ); 

% Open netCDF file, find variable IDs
ncId   = netcdf.open( fullfile( In.dir, files( 1 ).name ) );
idLon  = netcdf.inqVarID( ncId, 'lon' );
idLat  = netcdf.inqVarID( ncId, 'lat' );
idFld  = netcdf.inqVarID( ncId, In.var );

% Read longitude/latitude data, create region mask
lon  = netcdf.getVar( ncId, idLon );
lat  = netcdf.getVar( ncId, idLat );
nX  = length( lon );
nY  = length( lat );
[ X, Y ] = ndgrid( lon, lat );
rngMin = netcdf.getAtt( ncId, idFld, 'valid_min' );
rngMax = netcdf.getAtt( ncId, idFld, 'valid_max' );

% Close currently open netCDF file
netcdf.close( ncId );
 
% Output directory
dataDir = fullfile( Out.dir, ...
                    fldStr, ...
                    [ sprintf( 'x%i-%i', Domain.xLim ) ...
                      sprintf( '_y%i-%i', Domain.yLim ) ...
                      '_' Time.tLim{ 1 } '-' Time.tLim{ 2 } ] );
if ~isdir( dataDir )
    mkdir( dataDir )
end

% Prepare local and global indices for data retrieval
startNum = datenum( Time.tStart, Time.tFormat );
limNum = datenum( Time.tLim, Time.tFormat );
nT = months( limNum( 1 ), limNum( 2 ) ) + 1;
idxT1 = months( startNum, limNum( 1 ) ) + 1;   % first global index
idxTEnd = months( startNum, limNum( 2 ) ) + 1; % last global index
idxFile1 = findBatch( partitionT, idxT1 );       % first file
idxFileEnd = findBatch( partitionT, idxTEnd );   % last file
iTRead1 = idxT1 + 1 - getLowerBatchLimit( partitionT, idxFile1 ); 
iTReadEnd = idxTEnd + 1 - getLowerBatchLimit( partitionT, idxFileEnd );

% Prepare local and global indices for climatology
if ifClim
    climNum = datenum( Time.tClim, Time.tFormat );
    nTClim = months( climNum( 1 ), climNum( 2 ) ) + 1;
    idxTClim1 = months( startNum, climNum( 1 ) ) + 1; 
    idxTClimEnd = months( startNum, climNum( 2 ) ) + 1;
    idxFileClim1 = findBatch( partitionT, idxTClim1 );
    idxFileClimEnd = findBatch( partitionT, idxTClimEnd );
    iTClimRead1 = idxTClim1 + 1 ...
                     - getLowerBatchLimit( partitionT, idxFileClim1 ); 
    iTClimReadEnd = idxTClimEnd + 1 ...
                     - getLowerBatchLimit( partitionT, idxFileClimEnd );
end

% Loop over the netCDF files, read data
iFiles = idxFile1 : idxFileEnd;
nTReads = sum( nTFiles( iFiles ) );
if ifClim
    iFiles = unique( [ iFiles idxFileClim1 : idxFileClimEnd ] ); 
end
fldRead = zeros( nX, nY, nTReads ); 
iT1 = 1;
for iFile = iFiles
    
    fileIn = fullfile( In.dir, files( iFile ).name );
    %disp( sprintf( 'Extracting data from file %s', fileIn ) )
    ncId   = netcdf.open( fileIn  );

    % Number of samples in current file
    nTFile = nTFiles( iFile ); 

    % Read data from netCDF file
    iT2 = iT1 + nTFile - 1; 
    fldRead( :, :, iT1 : iT2 ) = netcdf.getVar( ncId, idFld );
    netcdf.close( ncId ); 
    iT1 = iT2 + 1; 

end


% Create region mask. Here, we are being conservative and
% only retain grid points with physical values for the entire temporal
% extent of the dataset. 
%fldRef = netcdf.getVar( ncId, idFld, [ 0 0 0 ], [ nX nY 1 ] ); 
%ifXY = X >= Domain.xLim( 1 ) & X <= Domain.xLim( 2 ) ...
%     & Y >= Domain.yLim( 1 ) & Y <= Domain.yLim( 2 ) ...
%     & fldRef >= rng( 1 ) & fldRef <= rng( 2 );
ifXY = X >= Domain.xLim( 1 ) & X <= Domain.xLim( 2 ) ...
     & Y >= Domain.yLim( 1 ) & Y <= Domain.yLim( 2 ) ...
     & all( fldRead >= rngMin & fldRead <= rngMax, 3 );
iXY = find( ifXY( : ) );
iXY = find( ifXY( : ) );
nXY = length( iXY );

% Reshape read data into rank-2 array
fldRead = reshape( fldRead, [ nX * nY nTReads ] );

% Prepare output arrays
iT1 = 1;
jT1 = 1;
fld = zeros( nXY, nT );
if ifClim
    iTClim1 = 1;
    cliData = zeros( nXY, nTClim );
end


% Put unmasked data into output arrays
for iFile = iFiles
    
    % Number of samples in current file
    nTFile = nTFiles( iFile ); 
    jT2 = jT1 + nTFile - 1;

    % Number of samples to read into output data array
    if iFile >= idxFile1 && iFile < idxFileEnd
        nTRead = nTFile;
    elseif iFile == idxFileEnd
        nTRead = iTReadEnd - iTRead1 + 1;
    else
        nTRead = 0;
    end
        
    % Number of samples to read into climatology array
    if ifClim
        if iFile >= idxFileClim1 && iFile < idxFileClimEnd
            nTClimRead = nTFile;
        elseif iFile == idxFileClimEnd
            nTClimRead = iTClimReadEnd - iTClimRead1 + 1;
        else
            nTClimRead = 0;
        end
    end

    % Read data into output data array, update time indices 
    if nTRead > 0 
        iT2 = iT1 + nTRead - 1;
        fld( :, iT1 : iT2 ) = fldRead( iXY, jT1 : jT1 + nTRead - 1 );
        iT1 = iT2 + 1;
    end

    % Read data into climatology array, update time indices
    if ifClim && nTClimRead > 0
        iTClim2 = iTClim1 + nTClimRead - 1;
        iTClimRead2 = iTClimRead1 + nTClimRead - 1;
        cliData( :, iTClim1 : iTClim2 ) = ...
            fldRead( iXY, jT1 : jT1 + nTClimRead - 1 );
        iTClim1 = iTClim2 + 1;
    end

    jT1 = jT2 + 1;
end

% If requested, weigh the data by the (normalized) grid cell surface areas. 
% Surface area calculation is approximate as it treats Earth as spherical
if Opts.ifWeight

    % Convert to radians and augment grid periodically 
    resLon = lon( 2 ) - lon( 1 );
    resLat = lat( 2 ) - lat( 1 );
    dLon = [ lon( 1 ) - resLon; lon; lon( end ) + resLon ] * pi / 180; 
    dLat = [ lat( 1 ) - resLat; lat; lat( end ) + resLat ] * pi / 180;

    % Compute grid coordinate differences
    dLon = ( dLon( 1 : end - 1 ) + dLon( 2 : end ) ) / 2;
    dLon = abs( dLon( 2 : end ) - dLon( 1 : end - 1 ) );
    dLat = ( dLat( 1 : end - 1 ) + dLat( 2 :end ) ) / 2;
    dLat = dLat( 2 : end ) - dLat( 1 : end - 1 );
    dLat = abs( dLat .* cos( lat * pi / 180 ) );

    % Compute surface area weights
    w = dLon .* dLat';
    w = w( ifXY );
    w = sqrt( w / sum( w ) * nXY );
      
    disp( sprintf( '%1.3g max area weight', max( w ) ) )
    disp( sprintf( '%1.3g min area weight', min( w ) ) )

    % Weigh the data
    fld = fld .* w;
end



% If requested, subtract global climatology
if Opts.ifCenter
    cli = mean( cliData, 2 );
    fld = fld - cli;
end

% If requested, subtract monthly climatology
if Opts.ifCenterMonth
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

% If requested, perform linear detrending
% beta is an [ nXY 2 ]-sized array such that b( i, 1 ) and b( i, 2 ) contain
% the mean and linear trend coefficients of the data. 
if Opts.ifDetrend
    t = [ 0 : nT - 1 ] / 12; % time in years
    beta = zeros( nXY, 2 ); 
    for j = 1 : nXY
        p = polyfit( t, fld( j, : ), 1 );
        beta( j, : ) = p;
    end
    fld = fld - beta( :, 2 ) - beta( :, 1 ) * t;
end 

    
% If requested, perform area averaging
if Opts.ifAverage
    fld = mean( fld, 1 );
    if ifClim
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
if Opts.ifWeight
    gridVarList = [ gridVarList 'w' ];
end
if Opts.ifWrite
    gridFile = fullfile( dataDir, 'dataGrid.mat' );
    save( gridFile, gridVarList{ : }, '-v7.3' )  
end

% Output data and attributes
x = double( fld ); % for compatibility with NLSA code
varList = { 'x' };
if Opts.ifCenter || Opts.ifCenterMonth
    varList = [ varList 'cli' 'nTClim' ];
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
    if ~Opts.ifOutputData
        % Exclude data from output 
        varList = varList( 2 : end );
    end
    nVar = numel( varList );
    vars = cell( 1, nVar );
    for iVar = 1 : nVar
       vars{ iVar } = eval( varList{ iVar } );
    end
    Data = cell2struct( vars, varList, 2 );
end


