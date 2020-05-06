% Script to add convective and large scale precipitation data from CCSM4
% control

%dataDir = '/Volumes/TooMuch/physics/climate/data/ccsm4/b40.1850'; 
dataDir = '/kontiki_array5/data/ccsm4/b40.1850'; 

fileBase = 'b40.1850.track1.1deg.006.cam2.h0.';

% Input/output variables (we add var1 to var2)
var1   = 'PRECC';
var2   = 'PRECL';
varOut = 'PREC'; 

% Input/output directoties
dir1 = fullfile( dataDir, var1 );
dir2 = fullfile( dataDir, var2 );
dirOut = fullfile( dataDir, varOut );

if ~isdir( dirOut )
    mkdir( dirOut )
end

files1 = dir( fullfile( dir1, [ fileBase '*.nc' ] ) );
files2 = dir( fullfile( dir2, [ fileBase '*.nc' ] ) );

nFileBase = numel( fileBase );
nVar1     = numel( var1 );
nVar2     = numel( var2 );

for iFile = 1 : numel( files1 )
    tic

    file1 = fullfile( dir1, files1( iFile ).name );
    file2 = fullfile( dir2, files2( iFile ).name );
    fileOut = [ files1( iFile ).name( 1 : nFileBase ) ...
                varOut ...
                files1( iFile ).name( nFileBase + nVar1 + 1 : end ) ];
    fileOut = fullfile( dirOut, fileOut );
    fileTmp = fullfile( dirOut, 'tmp.nc' );

    disp( 'Merging files...' )
    disp( file1 )
    disp( file2 )
    copyfile( file1, fileTmp )
    ncCommand = [ 'ncks -A ' file2 ' ' fileTmp ];
    disp( [ 'Command: ' ncCommand ] )
    system( ncCommand );


    disp( 'Adding data in merged file file...' )
    disp( fileOut )
    ncCommand = [ 'ncap2 -s "' varOut '=' var1 '+' var2 '" -v ' fileTmp ...
                  ' ' fileOut ];
    disp( [ 'Command: ' ncCommand ] )
    system( ncCommand );
    delete( fileTmp )

    toc
end





