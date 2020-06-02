function demoEnsoLifecycle_data( dataset, period, fld )
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
% Modified 2020/06/01

switch( dataset )

%% ERSSTv5 AND OTHER NOAA REANALYSIS PRODUCTS
case 'ersstV5'

    % Input data directory 
    %DataSpecs.In.dir  = '/Volumes/TooMuch/physics/climate/data/noaa'; 
    DataSpecs.In.dir = '/Users/dg227/GoogleDrive/physics/climate/data'; 

    % Output data specification
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );

    % Time specification
    DataSpecs.Time.tFormat = 'yyyymm';              % time format
    switch( period )

    % Satellite era
    case 'satellite'

        DataSpecs.Time.tLim    = { '197801' '202003' }; % time limits
        DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology 

    % Last 50 years
    case '50yr'

        DataSpecs.Time.tLim    = { '197001' '202003' }; % time limits
        DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology 


    otherwise

        error( 'Invalid period' )
    end

    switch( fld )

    %% Indo-Pacific SST (from ERSST)
    case 'globalSST'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'ersst.v5'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '188001';           % start time in nc file 

        % Spatial domain
        DataSpecs.Domain.xLim = [ 0 359 ]; % longitude limits
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits
    
        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = false; % don't remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV5( DataSpecs )

    %% Indo-Pacific SST (from ERSST)
    case 'IPSST'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'ersst.v5'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '188001';           % start time in nc file 

        % Spatial domain
        DataSpecs.Domain.xLim = [ 28 290 ]; % longitude limits
        DataSpecs.Domain.yLim = [ -60 20 ]; % latitude limits
    
        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = false; % don't remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV5( DataSpecs )

    %% Nino 4 index (from ERSST)
    case 'Nino4'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'ersst.v5'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '188001';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 160 210 ]; % longitude limits 
        DataSpecs.Domain.yLim = [ -5 5 ];    % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV5( DataSpecs )

    %% Nino 3.4 index (from ERSST)
    case 'Nino3.4'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'ersst.v5'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '188001';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 190 240 ]; % longitude limits 
        DataSpecs.Domain.yLim = [ -5 5 ];    % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV5( DataSpecs )

    %% Nino 3 index (from ERSST)
    case 'Nino3'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'ersst.v5'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '188001';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 210 270 ]; % longitude limits 
        DataSpecs.Domain.yLim = [ -5 5 ];    % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV5( DataSpecs )

    %% Nino 1+2 index (from ERSST)
    case 'Nino1+2'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'ersst.v5'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '188001';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 270 280 ]; % longitude limits 
        DataSpecs.Domain.yLim = [ -10 0 ];    % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV5( DataSpecs )

    %% Global SST (from ERSST)
    case 'SST' 

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'ersst.v5'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '188001';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV5( DataSpecs )

    % Global SSH (from GODAS)
    case 'SSH'

        % Input data 
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'godas' );
        DataSpecs.In.file = 'sshg'; % input filename
        DataSpecs.In.var  = 'sshg';

        % Output data
        DataSpecs.Out.fld = 'ssh';      

        % Time specification
        DataSpecs.Time.tStart = '198001'; % start time in dataset
        DataSpecs.Time.tEnd   = '202004'; % end time in dataset

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_godas( DataSpecs )

    %% Global SAT (from NCEP/NCAR)
    case( 'SAT' )

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ncep' );
        DataSpecs.In.file = 'air.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'air';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '194801';  % start time in dataset

        % Spatial domain 
        DataSpecs.Domain.xLim   = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim   = [ -89 89 ]; % latitude limits
        DataSpecs.Domain.levels = 1;          % levels  

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk


        importData_ncep( DataSpecs ) 

    %% Global precipitation rate (from CMAP)
    case 'precip'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'cmap' );
        DataSpecs.In.file = 'precip.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'precip';

        % Output data
        DataSpecs.Out.fld = 'prate';

        % Time specification
        DataSpecs.Time.tStart  = '197901'; % start time in dataset

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_cmap( DataSpecs )

    %% Global zonal wind (from NCEP/NCAR)
    case 'uwind' 

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ncep' );
        DataSpecs.In.file = 'uwnd.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'uwnd';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart = '194801'; % start time in dataset

        % Spatial domain 
        DataSpecs.Domain.xLim   = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim   = [ -89 89 ]; % latitude limits
        DataSpecs.Domain.levels = 1;          % levels  

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ncep( DataSpecs ) 

    %% Global meridional wind (from NCEP/NCAR)
    case 'vwind' 

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ncep' );
        DataSpecs.In.file = 'vwnd.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'vwnd';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '194801'; % start time in dataset 

        % Spatial domain 
        DataSpecs.Domain.xLim   = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim   = [ -89 89 ]; % latitude limits
        DataSpecs.Domain.levels = 1;          % levels  

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ncep( DataSpecs ) 

    end


%% ERSSTv4 AND OTHER NOAA REANALYSIS PRODUCTS
case 'ersstV4'

    % Input data directory 
    %DataSpecs.In.dir  = '/Volumes/TooMuch/physics/climate/data/noaa'; 
    DataSpecs.In.dir = '/Users/dg227/GoogleDrive/physics/climate/data'; 

    % Output data specification
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );

    % Time specification
    DataSpecs.Time.tFormat = 'yyyymm';              % time format
    switch( period )

    % Satellite era
    case 'satellite'

        DataSpecs.Time.tLim    = { '197801' '202003' }; % time limits
        DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology 

    % Last 50 years
    case '50yr'

        DataSpecs.Time.tLim    = { '197001' '202003' }; % time limits
        DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology 


    otherwise

        error( 'Invalid period' )
    end

    switch( fld )

    %% Indo-Pacific SST (from ERSST)
    case 'globalSST'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v4' );
        DataSpecs.In.file = 'sst.mnmean.v4.nc'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '185401';           % start time in nc file 

        % Spatial domain
        DataSpecs.Domain.xLim = [ 0 359 ]; % longitude limits
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits
    
        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = false; % don't remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV4( DataSpecs )

    %% Indo-Pacific SST (from ERSST)
    case 'IPSST'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v4' );
        DataSpecs.In.file = 'st.mnmean.v4.nc'; % input filename
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
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = false; % don't remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV4( DataSpecs )

    %% Nino 4 index (from ERSST)
    case 'Nino4'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'sst.mnmean.v4.nc'; % input filename
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
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV4( DataSpecs )

    %% Nino 3.4 index (from ERSST)
    case 'Nino3.4'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'sst.mnmean.v4.nc'; % input filename
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
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV4( DataSpecs )

    %% Nino 3 index (from ERSST)
    case 'Nino3'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'sst.mnmean.v4.nc'; % input filename
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
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV4( DataSpecs )

    %% Nino 1+2 index (from ERSST)
    case 'Nino1+2'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'sst.mnmean.v4.nc'; % input filename
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
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = true;  % perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV4( DataSpecs )

    %% Global SST (from ERSST)
    case 'SST' 

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v5' );
        DataSpecs.In.file = 'ersst.v5'; % input filename
        DataSpecs.In.var  = 'sst';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '188001';           % start time in nc file 

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV4( DataSpecs )

    % Global SSH (from GODAS)
    case 'SSH'

        % Input data 
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'godas' );
        DataSpecs.In.file = 'sshg'; % input filename
        DataSpecs.In.var  = 'sshg';

        % Output data
        DataSpecs.Out.fld = 'ssh';      

        % Time specification
        DataSpecs.Time.tStart = '198001'; % start time in dataset
        DataSpecs.Time.tEnd   = '202004'; % end time in dataset

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_godas( DataSpecs )

    %% Global SAT (from NCEP/NCAR)
    case( 'SAT' )

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ncep' );
        DataSpecs.In.file = 'air.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'air';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '194801';  % start time in dataset

        % Spatial domain 
        DataSpecs.Domain.xLim   = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim   = [ -89 89 ]; % latitude limits
        DataSpecs.Domain.levels = 1;          % levels  

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk


        importData_ncep( DataSpecs ) 

    %% Global precipitation rate (from CMAP)
    case 'precip'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'cmap' );
        DataSpecs.In.file = 'precip.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'precip';

        % Output data
        DataSpecs.Out.fld = 'prate';

        % Time specification
        DataSpecs.Time.tStart  = '197901'; % start time in dataset

        % Spatial domain 
        DataSpecs.Domain.xLim = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_cmap( DataSpecs )

    %% Global zonal wind (from NCEP/NCAR)
    case 'uwind' 

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ncep' );
        DataSpecs.In.file = 'uwnd.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'uwnd';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart = '194801'; % start time in dataset

        % Spatial domain 
        DataSpecs.Domain.xLim   = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim   = [ -89 89 ]; % latitude limits
        DataSpecs.Domain.levels = 1;          % levels  

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ncep( DataSpecs ) 

    %% Global meridional wind (from NCEP/NCAR)
    case 'vwind' 

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ncep' );
        DataSpecs.In.file = 'vwnd.mon.mean.nc'; % input filename
        DataSpecs.In.var  = 'vwnd';

        % Output data
        DataSpecs.Out.fld = DataSpecs.In.var;      

        % Time specification
        DataSpecs.Time.tStart  = '194801'; % start time in dataset 

        % Spatial domain 
        DataSpecs.Domain.xLim   = [ 0 359 ];  % longitude limits 
        DataSpecs.Domain.yLim   = [ -89 89 ]; % latitude limits
        DataSpecs.Domain.levels = 1;          % levels  

        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ncep( DataSpecs ) 

    end





%% NOAA 20th CENTURY REANALYSIS
case '20CR'

    % Input data directory 
    DataSpecs.In.dir  = '/Volumes/TooMuch/physics/climate/data/noaa'; 

    % Output data specification
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );

    % Time specification
    DataSpecs.Time.tFormat = 'yyyymm';              % time format
    switch( period )

    % Industrial era
    case 'industrial' 

        DataSpecs.Time.tLim    = { '187001' '201906' }; % time limits
        DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology 

    % Satellite era
    case 'satellite'

        DataSpecs.Time.tLim    = { '197001' '201906' }; % time limits
        DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology 

    otherwise

        error( 'Invalid period' )
    end

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

        importData_ersstV4( DataSpecs )

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

        importData_ersstV4( DataSpecs )

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

        importData_ersstV4( DataSpecs )

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

        importData_ersstV4( DataSpecs )

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

        importData_ersstV4( DataSpecs )

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

        importData_ersstV4( DataSpecs )

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
        %DataSpecs.Time.tStart  = '187101';           % start time in nc file 
        DataSpecs.Time.tStart  = '188001';            % start time in nc file 

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

        importData_noaa( DataSpecs )

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

        importData_noaa( DataSpecs )

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

        importData_noaa( DataSpecs )

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

        importData_noaa( DataSpecs )
    end

%%CCSM4 PRE-INDUSTRIAL CONTROL RUN 
case 'ccsm4Ctrl'

    % Input data directory 
    %DataSpecs.In.dir  = '/Volumes/TooMuch/physics/climate/data/ccsm4/b40.1850'; 
    DataSpecs.In.dir = '/kontiki_array5/data/ccsm4/b40.1850';

    % Output data specification
    DataSpecs.Out.dir = fullfile( pwd, 'data/raw', dataset );

    % Time specification
    DataSpecs.Time.tFormat = 'yyyymm';              % time format
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
        error( 'Invalid period' )
    end


    switch( fld )

    %% Global SST
    case 'globalSST'

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.pop.h.SST'; 
        DataSpecs.In.lon  = 'TLONG';
        DataSpecs.In.lat  = 'TLAT';
        DataSpecs.In.area = 'TAREA';
        DataSpecs.In.msk  = 'REGION_MASK';
        DataSpecs.In.var  = 'SST';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'sst';      

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

        % Spatial domain
        DataSpecs.Domain.xLim = [ 0 359 ]; % longitude limits
        DataSpecs.Domain.yLim = [ -89 89 ]; % latitude limits
    
        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = false; % don't remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ccsm4Ctrl( DataSpecs ) 


    %% Indo-Pacific SST
    case 'IPSST'

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.pop.h.SST'; 
        DataSpecs.In.lon  = 'TLONG';
        DataSpecs.In.lat  = 'TLAT';
        DataSpecs.In.area = 'TAREA';
        DataSpecs.In.msk  = 'REGION_MASK';
        DataSpecs.In.var  = 'SST';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'sst';      

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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

        importData_ccsm4Ctrl( DataSpecs ) 

    %% Nino 4 index
    case 'Nino4'

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.pop.h.SST'; 
        DataSpecs.In.lon  = 'TLONG';
        DataSpecs.In.lat  = 'TLAT';
        DataSpecs.In.area = 'TAREA';
        DataSpecs.In.msk  = 'REGION_MASK';
        DataSpecs.In.var  = 'SST';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'sst';

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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

        importData_ccsm4Ctrl( DataSpecs ) 

    %% Nino 3.4 index
    case 'Nino3.4'

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.pop.h.SST'; 
        DataSpecs.In.lon  = 'TLONG';
        DataSpecs.In.lat  = 'TLAT';
        DataSpecs.In.area = 'TAREA';
        DataSpecs.In.msk  = 'REGION_MASK';
        DataSpecs.In.var  = 'SST';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'sst';

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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

        importData_ccsm4Ctrl( DataSpecs ) 

    %% Nino 3 index
    case 'Nino3'

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.pop.h.SST'; 
        DataSpecs.In.lon  = 'TLONG';
        DataSpecs.In.lat  = 'TLAT';
        DataSpecs.In.area = 'TAREA';
        DataSpecs.In.msk  = 'REGION_MASK';
        DataSpecs.In.var  = 'SST';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'sst';

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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

        importData_ccsm4Ctrl( DataSpecs ) 

    %% Nino 1+2 index
    case 'Nino1+2'

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.pop.h.SST'; 
        DataSpecs.In.lon  = 'TLONG';
        DataSpecs.In.lat  = 'TLAT';
        DataSpecs.In.area = 'TAREA';
        DataSpecs.In.msk  = 'REGION_MASK';
        DataSpecs.In.var  = 'SST';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'sst';

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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

        importData_ccsm4Ctrl( DataSpecs ) 

    %% Global SST
    case( 'SST' )

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.pop.h.SST'; 
        DataSpecs.In.lon  = 'TLONG';
        DataSpecs.In.lat  = 'TLAT';
        DataSpecs.In.area = 'TAREA';
        DataSpecs.In.msk  = 'REGION_MASK';
        DataSpecs.In.var  = 'SST';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'sst';

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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

        importData_ccsm4Ctrl( DataSpecs ) 

    %% Global SSH
    case( 'SSH' )

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.pop.h.SSH'; 
        DataSpecs.In.lon  = 'TLONG';
        DataSpecs.In.lat  = 'TLAT';
        DataSpecs.In.area = 'TAREA';
        DataSpecs.In.msk  = 'REGION_MASK';
        DataSpecs.In.var  = 'SSH';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'ssh';

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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


        importData_ccsm4Ctrl( dataset, DataSpecs ) 

    %% Global SAT
    case( 'SAT' )

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.cam2.h0.TS'; 
        DataSpecs.In.lon  = 'lon';
        DataSpecs.In.lat  = 'lat';
        DataSpecs.In.var  = 'TS';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 


        % Output data
        DataSpecs.Out.fld = 'air';      

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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


        importData_ccsm4Ctrl( DataSpecs ) 

    %% Global precipitation data
    case( 'precip' )

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.cam2.h0.PREC'; 
        DataSpecs.In.lon  = 'lon';
        DataSpecs.In.lat  = 'lat';
        DataSpecs.In.var  = 'PREC';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, DataSpecs.In.var ); 

        % Output data
        DataSpecs.Out.fld = 'prate';      

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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

        importData_ccsm4Ctrl( DataSpecs ) 

    %% Global zonal wind data
    case( 'uwind' )

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.cam2.h0.USurf'; 
        DataSpecs.In.lon  = 'lon';
        DataSpecs.In.lat  = 'lat';
        DataSpecs.In.var  = 'U';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'USurf' ); 

        % Output data
        DataSpecs.Out.fld = 'uwnd';      

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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

        importData_ccsm4Ctrl( DataSpecs ) 

    %% Global meridional wind data
    case( 'vwind' )

        % Input data
        DataSpecs.In.file = 'b40.1850.track1.1deg.006.cam2.h0.VSurf'; 
        DataSpecs.In.lon  = 'lon';
        DataSpecs.In.lat  = 'lat';
        DataSpecs.In.var  = 'V';
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'VSurf' ); 

        % Output data
        DataSpecs.Out.fld = 'vwnd';      

        % Time specification
        DataSpecs.Time.tStart  = '000101';           % start time in nc file 

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

        importData_ccsm4Ctrl( DataSpecs ) 


    end

end

