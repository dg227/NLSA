function spczRainfall_data( dataset, period, fld )
% DEMOENSOLIFECYCLE_DATA Helper function to import datasets for ENSO lifecycle
% analyses.
%
% dataset - String identifier for dataset to read. 
% period  - String identifier for time period. 
% fld     - String identifier for variable to read. 
%
% This function creates a data structure with input data specifications as 
% appropriate for the dataset and fld arguments. 
%
% The data is then retrieved and saved on disk using the climateData function. 
%
% Modified 2020/05/09

%% GLOBAL DATA SPECIFICATIONS

% Root directory where data is stored
DataSpecs.In.dir  = '/Volumes/TooMuch/physics/climate/data'; 
% DataSpecs.In.dir = '/kontiki_array5/data/ccsm4/b40.1850';


% Output data specification
DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );

% Time specification
DataSpecs.Time.tFormat = 'yyyymm';              % time format

switch dataset

%% NOAA/CMAP REANALYSIS DATA
case 'noaa'

    % Input data directory 
    DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'noaa' ); 

    % Time specification
    DataSpecs.Time.tStart  = '197901';           % start time in nc file 

    switch( period )

    % Satellite era
    case 'satellite' 

        DataSpecs.Time.tLim    = { '197901' '201912' }; % time limits
        DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology 

    otherwise
        error( 'Invalid period.' )
    end

    switch( fld )

    %% Indo-Pacific precipitation
    case 'IPPrecip'

        % Input data
        DataSpecs.In.file = 'CMAP_extended.nc'; 
        DataSpecs.In.lon  = 'lon';
        DataSpecs.In.lat  = 'lat';
        DataSpecs.In.var  = 'precip';

        % Output data
        DataSpecs.Out.fld = 'prate';      

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 28 290 ]; % longitude limits
        DataSpecs.Domain.yLim = [ -60 20 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = false;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    %% Pacific precipitation
    case 'PacPrecip'

        % Input data
        DataSpecs.In.file = 'CMAP_extended.nc'; 
        DataSpecs.In.lon  = 'lon';
        DataSpecs.In.lat  = 'lat';
        DataSpecs.In.var  = 'precip';

        % Output data
        DataSpecs.Out.fld = 'prate';      

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 135 270 ]; % longitude limits
        DataSpecs.Domain.yLim = [ -35 35 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = false;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    otherwise
        error( 'Invalid variable.' )

    end

%%CCSM4 PRE-INDUSTRIAL CONTROL RUN 
case 'ccsm4Ctrl'

    % Input data directory 
    DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ccsm4/b40.1850' ); 

    % Time specification
    DataSpecs.Time.tStart  = '000101';           % start time in nc file 

    switch( period )

    % Time period comparable to industrial era
    case '200yr' 

        DataSpecs.Time.tLim    = { '000101' '019912' }; % time limits
        DataSpecs.Time.tClim   = DataSpecs.Time.tLim;  % climatology 

    % Full 1300-yr control integration
    case '1300yr'

        DataSpecs.Time.tLim    = { '000101' '130012' }; % time limits
        DataSpecs.Time.tClim   = DataSpecs.Time.tLim;   % climatology 

    otherwise
        error( 'Invalid period.' )

    end


    switch( fld )

    %% Indo-Pacific precipitation
    case 'IPPrecip'

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.cam2.h0.PREC'; 
        DataSpecs.In.lon  = 'lon';
        DataSpecs.In.lat  = 'lat';
        DataSpecs.In.var  = 'PREC';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'prate';      

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 28 290 ]; % longitude limits
        DataSpecs.Domain.yLim = [ -60 20 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = false;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    %% Pacific precipitation
    case 'PacPrecip'

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.cam2.h0.PREC'; 
        DataSpecs.In.lon  = 'lon';
        DataSpecs.In.lat  = 'lat';
        DataSpecs.In.var  = 'PREC';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'prate';      

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 135 270 ]; % longitude limits
        DataSpecs.Domain.yLim = [ -35 35 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = false;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    otherwise

        error( 'Invalid variable.' )

    end

otherwise

    error( 'Invalid dataset.' )
end

% Read data
climateData( dataset, DataSpecs ) 

