%% FLOEDATA
% Script to read coarse Eulerian data from ice floe model and output it in
% suitable format for analysis using the NLSA code
%
% Available combinations of variables, stored in variable fld are:
% 'c':   sea ice concentration
% 'uv':  sea ice velocity
% 'uvo': ocean velocity
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
fileIn     = [ "quad_gyre_packed.mat" "quad_gyre_packed_ocn.mat" ];  
fld        = [ "c" "uvocn" ];  
experiment = 'quad_gyre_packed';
idxTLim    = [ 1 1500 ];

%% EXECUTION OPTIONS
ifCenter    = true; % remove climatology
ifNormalize = true; % normalize to unit L2 norm

%% CREATE OUTPUT STRING AND DIRECTORY 
nFld = numel( fld ); % number of input sources
nS   = idxTLim( 2 ) - idxTLim( 1 ) + 1; % number of samples

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
dataIn  = cell( 1, nFld );
nDIn    = zeros( 2, nFld ); %data space dimension for individual sources
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
    end

    % Read data
    Dat = load( fullfile( dirIn, fileIn( iFld ) ), varIn );
    Dat = getfield( Dat, varIn );
    Dat = Dat( idxIn, :, idxTLim( 1 ) : idxTLim( 2 ) );
    nDat = size( Dat );
    nDIn( 1 : 2, iFld ) = nDat( 1 : 2 );
    nDInTot( iFld ) = nDat( 1 ) * nDat( 2 );
    dataIn{ iFld } = reshape( Dat, nDInTot( iFld ), nS ); 
end

%% SUBTRACT TIME MEAN
if ifCenter
    cli = cell( 1, nFld ); % climatology
    for iFld = 1 : nFld
        cli{ iFld } = mean( dataIn{ iFld }, 2 );
        dataIn{ iFld } = bsxfun( @minus, dataIn{ iFld }, cli{ iFld } );
    end
end

%% NORMALIZE TO UNIT L2 NORM
if ifNormalize
    l2Norm = zeros( 1, nFld ); 
    for iFld = 1 : nFld
        l2Norm( iFld ) = sqrt( sum( dataIn{ iFld }( : ) .^ 2 ) ...
            / ( nDIn( 2, iFld ) * nS ) );
        dataIn{ iFld } = dataIn{ iFld } / l2Norm( iFld ); 
    end
end

%% CONCATENATE AND SAVE DATA, ATTRIBUTES
nD = sum( nDInTot );
x = zeros( nD, nS ); % variable name 'x' is for compatibility with NLSA code
idxX1 = 1;
for iFld = 1 : nFld
    idxX2 = idxX1 + nDInTot( iFld ) - 1;
    x( idxX1 : idxX2, : ) = dataIn{ iFld };
end

fldFile = fullfile( dataDir, 'dataX.mat' );
varList = { 'x' 'fld' 'nDIn' };
if ifCenter 
    varList = [ varList 'cli' ];
end
if ifNormalize
    varList = [ varList 'l2Norm' ];
end
save( fldFile, varList{ : },  '-v7.3' )  
