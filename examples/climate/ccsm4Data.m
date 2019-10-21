%% READ CCSM4 DATA AND OUUTPUT IN .MAT FORMAT APPROPRIATE FOR NLSA
%% CODE
%
% Longitude range is [ 0 359 ] at 1 degree increments
% Latitude range is [ -89 89 ] at 1 degree increments
%
% Period to compute climatology must start at a January
%
% Modified 2019/08/24

%dataDirIn  = '/kontiki_array5/data/ccsm4/b40.1850'; % directory name for input data
dataDirIn  = '/Volumes/TooMuch/physics/climate/data/ccsm4/b40.1850'; % directory name for input data
fileBase   = 'b40.1850.track1.1deg.006.pop.h.SST.'; % filename base for input data
fldIn      = 'SST';                 
experiment = 'ccsm4_b40.1850';      % label for data analysis experiment 
fld        = 'sst';                 % label for field 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Indo-Pacific domain
%xLim       = [ 28 290 ];            % longitude limits
%yLim       = [ -60  20  ];          % latitude limits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nino 3.4 region
%xLim = [ 190 240 ];                 % longitude limits 
%yLim = [ -5 5 ];                    % latitude limits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nino 1+2 region
%xLim = [ 270 280 ];                 % longitude limits 
%yLim = [ -10 0 ];                    % latitude limits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nino 3 region
%xLim = [ 210 270 ];                 % longitude limits 
%yLim = [ -5 5 ];                    % latitude limits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nino 4 region
xLim = [ 160 210 ];                 % longitude limits 
yLim = [ -5 5 ];                    % latitude limits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Full control integration
%tLim = { '000101' '130012' };
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hindcast
%tLim       = { '000101' '119912' }; % training time limits 
tLim      = { '120001' '121212' }; % verification time limits 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tClim      = { '000101' '130012' }; % time limits for climatology removal
tStart     = '000101';              % start time in nc file 
tFormat    = 'yyyymm';              % time format

ifCenter      = false;                 % remove climatology
ifWeight      = true;                 % perform area weighting
ifCenterMonth = false;                 % remove monthly climatology 
ifAverage     = true;                % perform area averaging
ifNormalize   = false;                 % normalize to unit L2 norm

% Check for consistency of climatological averaging
if ifCenter & ifCenterMonth
    error( 'Global and monthly climatology removal cannot be simultaneously selected' )
end

% Input data partition and filename 
yrBatch = 100; % nominal years per input file
nBatch = 13;
nYrBatch = ones( 1, 13 ) * yrBatch; % number of years in each batch
nYrBatch( 1 ) = nYrBatch( 1 ) - 1;
nYrBatch( 13 ) = nYrBatch( 13 ) + 1;
nTBatch = nYrBatch * 12; % number of monthly samples in each batch
partitionYr = nlsaPartition( 'idx', cumsum( nYrBatch ) ); % yearly partition
partitionT = nlsaPartition( 'idx', cumsum( nTBatch ) ); % monthly partition 


% Input data filenames
filenameIn = cell( 1, nBatch );
for iBatch = 1 : nBatch
    batchLim = getBatchLimit( partitionYr, iBatch );
    filenameIn{ iBatch } = fullfile( ...
                       dataDirIn,  ...
                       [ fileBase, ...
                         sprintf( '%04i01-%04i12.nc', batchLim( 1 ), batchLim( 2 ) ) ] );
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


% Open first netCDF file, determine variable IDs
ncId   = netcdf.open( filenameIn{ 1 } );
idLon  = netcdf.inqVarID( ncId, 'TLONG' );
idLat  = netcdf.inqVarID( ncId, 'TLAT' );
idArea = netcdf.inqVarID( ncId, 'TAREA' );
idMsk  = netcdf.inqVarID( ncId, 'REGION_MASK' );
idFld  = netcdf.inqVarID( ncId, fldIn );

% Read longitude/latitude data, create region mask
X    = netcdf.getVar( ncId, idLon );
Y    = netcdf.getVar( ncId, idLat );
ifXY = netcdf.getVar( ncId, idMsk ) ~= 0; % nonzero values are ocean gridpoints
ifXY = X >= xLim( 1 ) & X <= xLim( 2 ) ...
     & Y >= yLim( 1 ) & Y <= yLim( 2 ) ...
     & ifXY;
nXY = numel( ifXY );   % number of gridpoints 
iXY = find( ifXY( : ) ); % extract linear indices from area mask
nX = size( X, 1 );
nY = size( X, 2 );
nXY  = nnz( ifXY ); % number if gridpoints in mask
disp( sprintf( '%i unmasked gridpoints ', nXY ) )

% If needed, compute area weights
if ifWeight
    w = netcdf.getVar( ncId, idArea ); 
    w  = w( ifXY );
    w  = sqrt( w / sum( w ) * nXY );
    disp( sprintf( '%1.3g max area weight', max( w ) ) )
    disp( sprintf( '%1.3g min area weight', min( w ) ) )
end

% Close NetCDF file
netcdf.close( ncId );
 
% Prepare local and global indices for data retrieval
startNum = datenum( tStart, tFormat );
limNum = datenum( tLim, tFormat );
nT = months( limNum( 1 ), limNum( 2 ) ) + 1;
idxT0 = months( startNum, limNum( 1 ) );  
idxTEnd = months( startNum, limNum( 2 ) );
iBatch0 = findBatch( partitionT, idxT0 + 1 );
iBatchEnd = findBatch( partitionT, idxTEnd + 1 );
idxT0Loc = idxT0 + 1 - getLowerBatchLimit( partitionT, iBatch0 ); 
idxTEndLoc = idxTEnd + 1 - getLowerBatchLimit( partitionT, iBatchEnd );

% Retrieve data
fld = zeros( nXY, nT );
iT1 = 1;
for iBatch = iBatch0 : iBatchEnd
    disp( sprintf( 'Extracting data from file %s', filenameIn{ iBatch } ) )
    if iBatch < iBatchEnd
        nTRead = nTBatch( iBatch );
    else
        nTRead = idxTEndLoc - idxT0Loc + 1;
    end  
    iT2 = iT1 + nTRead - 1;
    ncId = netcdf.open( filenameIn{ iBatch } );
    fldBatch = netcdf.getVar( ncId, idFld, [ 0 0 0 idxT0Loc ], [ nX nY 1 nTRead ] );
    netcdf.close( ncId ); 
    fldBatch = reshape( fldBatch, [ nX * nY, nTRead ] );
    fld( :, iT1 : iT2 ) = fldBatch( iXY, : );
    iT1 = iT2 + 1;
    idxT0Loc = 0;
end

% If requested, weigh the data by the (normalized) grid cell surface areas. 
if ifWeight
    fld = bsxfun( @times, fld, w );
end

% If requested, read data for climatology
if ifCenter || ifCenterMonth

    % Prepare local and global indices for data retrieval
    climNum = datenum( tClim, tFormat );
    nTClim = months( climNum( 1 ), climNum( 2 ) ) + 1;
    idxTClim0 = months( startNum, climNum( 1 ) ); 
    idxTClimEnd = months( startNum, climNum( 2 ) );
    iBatchClim0 = findBatch( partitionT, idxTClim0 + 1 );
    iBatchClimEnd = findBatch( partitionT, idxTClimEnd + 1 );
    idxT0Loc = idxTClim0 + 1 - getLowerBatchLimit( partitionT, iBatchClim0 ); 
    idxTEndLoc = idxTClimEnd + 1 - getLowerBatchLimit( partitionT, iBatchClimEnd );

    % Retrieve data
    cliData = zeros( nXY, nTClim );
    iT1 = 1;
    for iBatch = iBatchClim0 : iBatchClimEnd
        disp( sprintf( 'Extracting data from file %s', filenameIn{ iBatch } ) )
        if iBatch < iBatchClimEnd
            nTRead = nTBatch( iBatch );
        else
            nTRead = idxTEndLoc - idxT0Loc + 1;
        end  
        iT2 = iT1 + nTRead - 1;
        ncId = netcdf.open( filenameIn{ iBatch } );
        fldBatch = netcdf.getVar( ncId, idFld, [ 0 0 0 idxT0Loc ], [ nX nY 1 nTRead ] );
        netcdf.close( ncId );
        fldBatch = reshape( fldBatch, [ nX * nY, nTRead ] );
        cliData( :, iT1 : iT2 ) = fldBatch( iXY, : );
        iT1 = iT2 + 1;
        idxT0Loc = 0;
    end

    % If requested, weigh the data
    if ifWeight
        cliData = bsxfun( @times, cliData, w );
    end
end

% If requested, subtract global climatology
if ifCenter
    cli = mean( cliData, 2 );
    fld = bsxfun( @minus, fld, cli );
end

% If requested, subtract monthly climatology
if ifCenterMonth
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
save( gridFile, 'X', 'Y', 'ifXY', 'fldStr', 'nD', '-v7.3' )  

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

