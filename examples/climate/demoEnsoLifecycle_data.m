function demoEnsoLifecycle_data( dataset, fld )
% DEMOENSOLIFECYCLE_DATA Helper function to import datasets for ENSO lifecycle
% analyses.
%
% dataset - String identifier for dataset to read. 
% fld     - String identifier for variable to read. 
%
% This function creates a data structure with input data specifications as 
% appropriate for the dataset and fld arguments. 
%
% The data is then retrieved and saved on disk using the climateData function. 
%
% Modified 2020/04/28

switch( dataset )

%% NOAA 20th CENTURY REANALYSIS
case 'noaa'

    % Input data directory 
    DataSpecs.In.dir  = '/Volumes/TooMuch/physics/climate/data/noaa'; 

    % Output data specification
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );

    % Time specification
    DataSpecs.Time.tLim    = { '187001' '201906' }; % time limits
    DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology time limits
    DataSpecs.Time.tFormat = 'yyyymm';              % time format

    switch( fld )

    %% Indo-Pacific SST
    case 'IPSST'

        % Input data
        DataSpecs.In.file = 'sst.mnmean.v4-4.nc'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185401';           % start time in nc file 

        % Spatial domain
        DataSpecs.Domain.xLim = [ 28 290 ]; % longitude limits
        DataSpecs.Domain.yLim = [ -60 20 ]; % latitude limits
    
        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = false; % don't remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    %% Nino 4 index
    case 'Nino4'

        % Input data
        DataSpecs.In.file = 'sst.mnmean.v4-4.nc'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185401';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 160 210 ]; % longitude limits 
        DataSpecs.Domain.yLim = [ -5 5 ];    % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk


    %% Nino 3.4 index
    case 'Nino3.4'

        % Input data
        DataSpecs.In.file = 'sst.mnmean.v4-4.nc'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185401';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 190 240 ]; % longitude limits 
        DataSpecs.Domain.yLim = [ -5 5 ];    % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    %% Nino 3 index
    case 'Nino3'

        % Input data
        DataSpecs.In.file = 'sst.mnmean.v4-4.nc'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185401';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 210 270 ]; % longitude limits 
        DataSpecs.Domain.yLim = [ -5 5 ];    % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    %% Nino 1+2 index
    case 'Nino1+2'

        % Input data
        DataSpecs.In.file = 'sst.mnmean.v4-4.nc'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185401';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 270 280 ]; % longitude limits 
        DataSpecs.Domain.yLim = [ -10 0 ];    % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    %% Global SST
    case( 'SST' )

        % Input data
        DataSpecs.In.file = 'sst.mnmean.v4-4.nc'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185401';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk


    %% Global SAT
    % Start dates/end dates for possible source files are as follows:
    %
    % air.2m.mon.mean-2.nc: 187101 to 201212
    % air.mon.anom.v5.nc:   188001 to 201908 
    case( 'SAT' )

        % Input data
        DataSpecs.In.file = 'air.2m.mon.mean-2.nc'; % input filename
        DataSpecs.In.var  = 'air';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '187101';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk


        climateData( dataset, DataSpecs ) % read global SAT data

    %% Global precipitation rate
    case( 'precip' )

        % Input data
        DataSpecs.In.file = 'prate.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'prate';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185101';              % start time in nc file 
        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    %% Global zonal wind
    % Start dates/end dates for possible source files are as follows:
    % 
    % uwnd.10m.mon.mean.nc: 185101 to 201412
    case( 'uwind' )

        % Input data
        DataSpecs.In.file = 'uwnd.10m.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'uwnd';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185101';              % start time in nc file 
        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    %% Global meridional wind
    % Start dates/end dates for possible source files are as follows:
    % 
    % vwnd.10m.mon.mean.nc: 185101 to 201412
    case( 'vwind' )

        % Input data
        DataSpecs.In.file = 'vwnd.10m.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'vwnd';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185101';              % start time in nc file 
        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

    end

end

% Read data
climateData( dataset, DataSpecs ) 
