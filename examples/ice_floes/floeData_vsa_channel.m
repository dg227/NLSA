%% FLOEDATA
% Script to read coarse Eulerian data from ice floe model and output it in
% suitable format for VSA using the NLSA code
%
% Available combinations of variables, stored in variable fld are:
% 'c':   sea ice concentration
% 'uv':  sea ice velocity
% 'uvo': ocean velocity
% 'acc': acceleration
%
% fld can be either a character string set to the above values, or a string
% array. In the latter case, the data vectors corresponding to the array 
% components are concatenated to form a single variable. 
%
% Additional parameters:
% idxTLim:    2-element vector specifying the time indices to be retrieved
% experiment: string identifier for the data analysis experiment
% fileIn:     input file for floe/ocean data
% pathIn:     input file directory

%% SCRIPT PARAMETERS
dirIn      = '../../..';
fileIn     = [ "EulCoarse.mat" ];  
fld        = [ "acc" ];  
experiment = 'channel';
idxTLim    = [ 1 200 ];
%idxTLim    = [ 201 250 ];

%% EXECUTION OPTIONS
ifCenter    = false; % remove climatology
ifNormalize = false;  % normalize to unit L2 norm

%% CREATE OUTPUT STRING AND DIRECTORY 
nFld = numel( fld ); % number of input sources

% Check and concatenate input data sources
if ischar( fld )
    fld = string( fld );
elseif ~isstring( fld )
    error( 'Requested data sources must be input as a character string or string vector' )
elseif ~isvector( fld )
    error( 'Requested data sources must be input as a string vector' )
end
    
fldStr = convertStringsToChars( strjoin( fld, '' ) );

% Append 'a' to field string if outputting anomalies
if ifCenter
    fldStr = [ fldStr 'a' ];
end

% Append 'n' if normalizing
if ifNormalize
    fldStr = [ fldStr 'n' ];
end

% Check input files
if ischar( fileIn )
    fileIn = repmat( string( fileIn ), 1, nFld );
elseif ~isstring( fileIn )
    error( 'Input filenames must be set to a character string or a string array' )
elseif ~isvector( fileIn ) || ~numel( fileIn ) == numel( fld ) 
    error( 'Input filenames must be set to a string vector' )
end
    
% Output directory
dataDir = fullfile( pwd, ...
                    'data/raw', ...
                    experiment, ...
                    fldStr, ...
                    sprintf( 'idxT_%i-%i', idxTLim( 1 ), idxTLim( 2 ) ) );
if ~isdir( dataDir )
    mkdir( dataDir )
end

%% READ DATA
dataIn = cell( 1, nFld );
nD     = zeros( 1, nFld );                % dimension for individual sources
nG     = [];                              % number of gridpoints 
nS     = idxTLim( 2 ) - idxTLim( 1 ) + 1; % number of samples

nDInTot = zeros( 1, nFld ); 
for iFld = 1 : nFld

    % Set variables to read from .mat file.
    % idxIn is the row from which data will be read
    switch fld( iFld )
    case 'c'
        idxIn = 1;
        varIn = 'EulCoarse';
    case 'uv'
        idxIn = [ 2 3 ];
        varIn = 'EulCoarse';
    case 'uvocn'
        idxIn = [ 1 2 ]; 
        varIn = 'EulCoarse_ocn';
    case 'acc'
        idxIn = [ 4 5 ];
        varIn = 'EulCoarse';
    end

    % Read data
    Dat = load( fullfile( dirIn, fileIn( iFld ) ), varIn );
    Dat = getfield( Dat, varIn );
    Dat = Dat( idxIn, :, idxTLim( 1 ) : idxTLim( 2 ) );
    
    % Assign dimension and number of gridpoints
    nD( iFld ) = size( Dat, 1 );
    if iFld == 1
        nG = size( Dat, 2 );
    elseif nG ~= size( Dat, 2 )
        error( 'Incompatible number of gridpoints for input source %i', iFld )
    end
    dataIn{ iFld } = reshape( Dat, nD( iFld ), nG * nS ); 
end

%% SUBTRACT TIME MEAN
if ifCenter
    cliIn = cell( 1, nFld ); % climatology
    for iFld = 1 : nFld
        cliIn{ iFld } = mean( dataIn{ iFld }, 2 );
        dataIn{ iFld } = bsxfun( @minus, dataIn{ iFld }, cliIn{ iFld } );
    end
end

%% NORMALIZE TO UNIT L2 NORM
if ifNormalize
    l2Norm = zeros( 1, nFld ); 
    for iFld = 1 : nFld
        l2Norm( iFld ) = sqrt( mean( dataIn{ iFld }.^ 2, 2 ) ); 
        dataIn{ iFld } = dataIn{ iFld } / l2Norm( iFld ); 
    end
end

%% CONCATENATE AND SAVE DATA, ATTRIBUTES
% xAll contains the full spatiotemporal data
% x contains gridpoint values
nDTot = sum( nD );
xAll = zeros( nDTot, nG * nS ); 
idxX1 = 1;
for iFld = 1 : nFld
    idxX2 = idxX1 + nD( iFld ) - 1;
    xAll( idxX1 : idxX2, : ) = dataIn{ iFld };
end
xAll = reshape( xAll, [ nDTot nG nS ] );
varList = { 'x' 'fld' 'nD' 'nDTot' 'nG' 'iG' };
if ifCenter 
    varList = [ varList 'cli' ];
end
if ifNormalize
    varList = [ varList 'l2Norm' ];
end
for iG = 1 : nG
    % reshaping to ensure that we output a row vector if nDTot == 1
    x = reshape( squeeze( xAll( :, iG, : ) ), [ nDTot nS ] );
    if ifCenter
        cli = cliIn{ iG };
    end
    fldFile = fullfile( dataDir, sprintf( 'dataX_%i.mat', iG ) );
    save( fldFile, varList{ : },  '-v7.3' )  
end


