% Script to extract surface wind data from CCSM4 control wind data
%
% Modified 2020/05/11

dataDir = '/Volumes/TooMuch/physics/climate/data/ccsm4/b40.1850'; 
%dataDir = '/kontiki_array5/data/ccsm4/b40.1850'; 

fileBase = 'b40.1850.track1.1deg.006.cam2.h0.';

% Input variables
%varIn    = 'U';
varIn    = 'V';

% Level to extract (surface)
levSurf = 26; 
levStr = int2str( levSurf );

% Output variables
varOut = [ varIn 'Surf' ];

% Input/output directoties
dirIn = fullfile( dataDir, varIn );
dirOut = fullfile( dataDir, varOut );

if ~isdir( dirOut )
    mkdir( dirOut )
end

filesIn = dir( fullfile( dirIn, [ fileBase '*.nc' ] ) );

nFileBase = numel( fileBase );
nVarIn    = numel( varIn );

for iFile = 1 : numel( filesIn )
    tic
    disp( 'Extracting surface wind data from file...' )
    fileIn = fullfile( dirIn, filesIn( iFile ).name );
    fileOut = [ filesIn( iFile ).name( 1 : nFileBase ) ...
                varOut ...
                filesIn( iFile ).name( nFileBase + nVarIn + 1 : end ) ];
    fileOut = fullfile( dirOut, fileOut );

    ncCommand = [ 'ncea -d lev,' levStr ',' levStr ...
                  ' -F ' fileIn ' ' fileOut ];
    disp( [ 'Command: ' ncCommand ] )
    system( ncCommand );

    toc
end





