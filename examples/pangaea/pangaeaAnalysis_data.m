function pangaeaAnalysis_data( fld )
% PANGAEA_ANALYSIS_DATA Helper function to import paleoclimate datasets from 
% PANGAEA
%
% fld     - String identifier for variable to read. 
%
% This function creates a data structure with input data specifications as 
% appropriate for the dataset and fld arguments. 
%
% The data is then retrieved and saved on disk using the pangaeaData function. 
%
% Modified 2020/05/09

% Input data directory 
DataSpecs.In.dir = '/Users/dg227/GoogleDrive/physics/climate/data/pangaea'; 

% Input data file
DataSpecs.In.file = 'Willeit_etal_2019.nc';

% Output data directory
DataSpecs.Out.dir = fullfile( pwd, 'data/raw' );

% Time specification
DataSpecs.Time.tLim = [ -3000 0 ];

switch( fld )

    %% Surface temperature 
    case 'temp'
        % Input data
        DataSpecs.In.var  = 'temp';

    otherwise 
        error( 'Invalid variable' )
end

% Output data
DataSpecs.Out.fld = DataSpecs.In.var;


% Output options
DataSpecs.Opts.ifCenter    = false; % don't remove global climatology
DataSpecs.Opts.ifNormalize = false; % don't normalize to unit L2 norm
DataSpecs.Opts.ifWrite     = true;  % write data to disk

importData_pangaea( DataSpecs )


