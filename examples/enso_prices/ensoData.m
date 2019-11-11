%% READ SPREADSHEET DATA AND OUUTPUT IN .MAT FORMAT APPROPRIATE FOR NLSA CODE
%
% Modified 2019/11/11
%tLim       = { '195801' '199512' }; % time limits to read (training) 
tLim       = { '199601' '199912' }; % time limits to read (verification) 
tClim      = { '195801' '199512' }; % time limits for climatology (optional)
tStart     = '195801';              % start time in data file 
tFormat    = 'yyyymm';              % time format
experiment = 'coffee';              % label for experiment
lbl        = 'price';               % label for data variable

ifCenter      = false;              % remove global climatology
ifCenterMonth = false;              % remove monthly climatology 
ifNormalize   = false;              % normalize to unit L2 norm

% Check for consistency of climatological averaging
if ifCenter & ifCenterMonth
    error( 'Global and monthly climatology removal cannot be simultaneously selected' )
end

lblStr = lbl; 

% Append 'a' to field string if outputting anomalies
if ifCenter 
    lblStr = [ lbl 'a' ];
end

% Append 'ma' to field string if outputting monthly anomalies
if ifCenterMonth
    lblStr = [ lbl 'ma' ];
end

% Append 'n' if normalizing
if ifNormalize
    lblStr = [ lblStr 'n' ];
end

% Append time limits for climatology 
if ifCenter | ifCenterMonth
    lblStr = [ lblStr '_' tClim{ 1 } '-' tClim{ 2 } ];
end 

% Output directory
dataDir = fullfile( pwd, ...
                    'data/raw', ...
                    experiment, ...
                    lblStr, ...
                    strjoin( tLim, '-' ) );
if ~isdir( dataDir )
    mkdir( dataDir )
end

% Number of samples and starting time indices
limNum = datenum( tLim, tFormat );
startNum = datenum( tStart, tFormat );
nT    = months( limNum( 1 ), limNum( 2 ) ) + 1;
idxT0 = months( startNum, limNum( 1 ) ) + 1;  

% Read data from spreadsheet
T = readtable('data2.xls');
T = T(:,2:5);
A = table2array(T)';

% Output data
x = A( :, idxT0 : idxT0 + nT - 1 );
nD = size( x, 1 ); % data dimension

% Data for climatology
if ifCenter || ifCenterMonth
    cliData = A( :, idxTClim0 : idxTClim0 + nTClim - 1 );
end

% If requested, subtract climatology
if ifCenter
    climNum = datenum( tClim, tFormat );
    nTClim = months( climNum( 1 ), climNum( 2 ) ) + 1;
    idxTClim0 = months( startNum, climNum( 1 ) ) + 1; 
    cli = mean( cliData,  2 );
    x = bsxfun( @minus, x, cli );
end

% If requested, subtract mnthly climatology
if ifCenterMonth
    cli = zeros( nD, 12 );
    for iM = 1 : 12
        cli( :, iM ) = mean( cliData( :, iM : 12 : end ), 2 );
    end
    idxM0 = month( limNum( 1 ) ); 
    for iM = 1 : 12
        idxM = mod( idxM0 + iM - 2, 12 ) + 1; 
        x( :, iM : 12 : end ) = bsxfun( @minus, x( :,  iM : 12 : end ), ...
                                                cli( :, idxM ) ); 
    end  
end


% If requested, normalize by RMS climatology norm
if ifNormalize
    l2Norm = norm( cli( : ), 2 ) / sqrt( nTClim );
    fld = fld / l2Norm;
end


% Save output data and attributes
dataFile = fullfile( dataDir, 'dataX.mat' );
varList = { 'x' 'idxT0' 'nD' };
if ifCenter || ifCenterMonth
    varList = [ varList 'cli' 'idxTClim0' 'nTClim' ];
end
if ifNormalize
    varList = [ varList 'l2Norm' ];
end
save( dataFile, varList{ : },  '-v7.3' )  

