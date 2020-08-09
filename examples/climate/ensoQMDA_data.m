function ensoQMDA_data( dataset, period, fld )
% ENSOQMDA_DATA Helper function to import datasets for quantum mechanical data
% assimilation of ENSO.
%
% dataset - String identifier for dataset to read. 
% period  - String identifier for time period. 
% fld     - String identifier for variable to read. 
%
% This function creates a parameter structure with input data specifications 
% as appropriate for the dataset and fld arguments. 
%
% The data is then retrieved and saved on disk using the climateData function. 
%
% Modified 2020/08/08

switch dataset 

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

    % 1960-2010
    case '50yr'

        DataSpecs.Time.tLim    = { '196001' '200912' }; % time limits
        DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology 

    % 1940-2010
    case '70yr'

        DataSpecs.Time.tLim    = { '194001' '200912' }; % time limits
        DataSpecs.Time.tClim   = { '198101' '201012' }; % climatology 


    otherwise

        error( 'Invalid period' )
    end

    switch( fld )

    %% Global SST (from ERSST)
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

    %% Sub-global SST (excludes high latitudes with noisy data)
    case 'subglobalSST'

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
        DataSpecs.Domain.yLim = [ -67 67 ]; % latitude limits
    
        % Output options
        DataSpecs.Opts.ifCenter      = false; % don't remove global climatology
        DataSpecs.Opts.ifDetrend     = false; % don't detrend
        DataSpecs.Opts.ifWeight      = true;  % perform area weighting
        DataSpecs.Opts.ifCenterMonth = false; % don't remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV4( DataSpecs )


    %% Indo-Pacific SST 
    case 'IPSST'

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v4' );
        DataSpecs.In.file = 'sst.mnmean.v4.nc'; % input filename
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
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v4' );
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
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v4' );
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
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v4' );
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
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v4' );
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

    %% Global SST anomalies relative to monthly climatology
    case 'SST' 

        % Input data
        DataSpecs.In.dir  = fullfile( DataSpecs.In.dir, 'ersst.v4' );
        DataSpecs.In.file = 'sst.mnmean.v4.nc'; % input filename
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
        DataSpecs.Opts.ifDetrend     = false;  % perform linear detrending
        DataSpecs.Opts.ifWeight      = false; % don't perform area weighting
        DataSpecs.Opts.ifCenterMonth = true;  % remove monthly climatology 
        DataSpecs.Opts.ifAverage     = false; % don't perform area averaging
        DataSpecs.Opts.ifNormalize   = false; % don't normalize to unit L2 norm
        DataSpecs.Opts.ifWrite       = true;  % write data to disk

        importData_ersstV4( DataSpecs )

    % Global SSH anomalies relative to monthly climatology (from GODAS)
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

    %% Global SAT anomalies relative to monthly climatology (from NCEP/NCAR)
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

    %% Global precipitation rate anomalies (from CMAP)
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

    %% Global zonal wind anomalies (from NCEP/NCAR)
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

    %% Global meridional wind anomalies (from NCEP/NCAR)
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

    otherwise
        error( 'Invalid variable' )

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

    % 1200 year period
    case '1200yr'

        DataSpecs.Time.tLim    = { '000101' '119912' }; % time limits
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

    %% Global SST anomalies relative to monthly climatology
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

    %% Global SSH anomalies relative to monthly climatology
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

    %% Global SAT anomalies relative to monthly climatology
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

    %% Global precipitation anomalies relative to monthly climatology
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

    %% Global zonal wind anomalies relative to monthly climatology
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

    %% Global meridional wind anomalies relative to monthly climatology
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


    otherwise
        error( 'Invalid variable' )
    end

otherwise
    error( 'Invalid dataset.' )
end

