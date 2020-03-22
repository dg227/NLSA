%% Script to read AL index and vBz (solar wind) data
%
% Outputs the data columnwise in an array x. 
%
% The data are standardized to zero mean and unit variance.  
%
% Dimitris Giannakis, dimitris@cims.nyu.edu
% 
% Modified 2017/04/15

%% SCRIPT PARAMETERS
yr     = 2000;         % year, either 2000 or 2003 
tLim   = [ 1 1000000 ];% timestamps to read (lower and upper limits), in minutes
nSkip  = 5;            % skip every nSkip timestamps 
nShift = 50;            % shift AL data to the future of the solar wind data
fld    = 'AL_SW';      % extract both AL and solar wind data
flgBz  = inf;          % flag values for missing Bz data; set to inf to ignore
flgV   = inf;          % flag values for missing v data; set to inf to ignore

%% IMPORT DATA
% Read AL index data from ASCII file
fileName = sprintf( 'SW_AL_%i.dat', yr );
data = dlmread( fileName );
dataAL = data( ( tLim( 1 ) : nSkip : tLim( 2 ) ) + nShift, 7 );

% Standardize the data
muAL = mean( dataAL );
dataAL = dataAL - muAL;
sigmaAL = sqrt( mean( dataAL .^ 2 ) );
dataAL = dataAL / sigmaAL;

switch fld
    case 'AL'
        x = dataAL';

    case 'AL_SW'
        % Read vBz data from ASCII file, standardize
        dataBz =  data( tLim( 1 ) : nSkip : tLim( 2 ), 5 );
        ifFlgBz = dataBz > flgBz;
        muBz = mean( dataBz( ~ifFlgBz ) );
        dataBz( ifFlgBz ) = muBz; % set missing data to mean
        dataV  =  data( tLim( 1 ) : nSkip : tLim( 2 ), 6 );
        ifFlgV = dataV > flgV;
        muV = mean( dataV( ~ifFlgV ) );
        dataV( ifFlgV ) = muV; % set missing data to mean
        dataSW = dataBz .* dataV;
        muSW = mean( dataSW );
        dataSW = dataSW - muSW;
        sigmaSW = sqrt( mean( dataSW .^ 2 ) );
        dataSW = dataSW / sigmaSW;
        x = [ dataAL'; dataSW' ];
end

%% WRITE DATA TO DISK
dirNameOut = fullfile( 'data', 'raw', fld );
if ~isdir( dirNameOut )
    mkdir( dirNameOut )
end
fileNameOut = sprintf( 'dataX_%i-%i_nSkip%i_nShift%i.mat', tLim( 1 ), tLim( 2 ), nSkip, nShift );
save( fullfile( dirNameOut, fileNameOut ), 'x', 'muAL', 'sigmaAL', 'muSW', 'sigmaSW' )
