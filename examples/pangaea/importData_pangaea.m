function Data = importData_pangaea( DataSpecs )
% IMPORTDATA_PANGAEA Read PANGAEA data from netCDF files, and output in format 
% appropriate for NLSA code.
% 
% DataSpecs is a data structure containing the specifications of the data to
% be read. 
%
% Data is a data structure containing the data read and associated attributes.

% DataSpecs has the following fields:
%
% In.dir:             Input directory name
% In.file:            Input filename
% In.var:             Variable to be read
% Out.dir:            Output directory name
% Out.fld:            Output label 
% Time.tLim:          Time limits (in ka) 
% Time.tClim:         Time limits for climatology removal
% Opts.ifCenter:      Remove global climatology if true 
% Opts.ifNormalize:   Standardize data to unit L2 norm if true
% Opts.ifWrite:       Write data to disk
% Opts.ifOutputData:  Only data attributes are returned if set to false
%
% Modified 2020/07/27

%% UNPACK INPUT DATA STRUCTURE FOR CONVENIENCE
In     = DataSpecs.In;
Out    = DataSpecs.Out; 
Time   = DataSpecs.Time;
Opts   = DataSpecs.Opts;

%% READ DATA

% Determine if we need to compute climatology
ifClim = Opts.ifCenter || Opts.ifNormalize;

% Append 'a' to field string if outputting anomalies
if Opts.ifCenter
    fldStr = [ Out.fld 'a' ];
else
    fldStr = Out.fld;
end

% Append 'n' if normalizing
if Opts.ifNormalize
    fldStr = [ fldStr 'n' ];
end

% Append time limits for climatology 
if ifClim 
    fldStr = sprintf( '%s_%i-%i', fldStr, Time.tClim( 1 ), Time.tClim( 2 ) ); 
end 

% Output directory
dataDir = fullfile( Out.dir, fldStr, ...
                    sprintf( '%i-%i', Time.tLim( 1 ), Time.tLim( 2 ) ) );
if Opts.ifWrite && ~isdir( dataDir )
    mkdir( dataDir )
end

% Number of samples and starting time index
% Indices are based at 0 to conform with NetCDF library
startNum = -3000; 
nT       = Time.tLim( 2 ) - Time.tLim( 1 ) + 1;
idxT0    = Time.tLim( 1 ) - startNum; 
if ifClim
    nTClim = Time.tClim( 2 ) - Time.tClim( 1 ) + 1; 
    idxTClim0 = Time.tClim( 1 ) - startNum;  
end

% Input netCDF file
ncFile = fullfile( In.dir, In.file );
ncId  = netcdf.open( fullfile( In.dir, In.file ) );

% Retrieve data
idFld = netcdf.inqVarID( ncId, In.var );
fld   = netcdf.getVar( ncId, idFld, idxT0, nT )';


% If needed, compute climatology
if ifClim
    fld   = netcdf.getVar( ncId, idFld, idxTClim0, nTClim )';
    cli = mean( cli );
end

% If requested, subtract climatology.
if Opts.ifCenter
    fld = fld - cli;
end

% If requested, normalize by RMS climatology norm
if Opts.ifNormalize
    l2Norm = norm( cli( : ), 2 ) / sqrt( nTClim );
    fld = fld / l2Norm;
end

% Close netCDF file
netcdf.close( ncId );

%% RETURN AND WRITE DATA

% Output data and attributes
x = double( fld ); % for compatibility with NLSA code
varList = { 'x' 'idxT0' };
if Opts.ifCenter
    varList = [ varList 'cli' 'idxTClim0' 'nTClim' ];
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
