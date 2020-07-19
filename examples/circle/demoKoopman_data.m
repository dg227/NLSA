function demoNLSA_data( experiment )
% DEMONLSA_DATA Helper function to generate datasets for NLSA demo for 
% variable-speed circle rotation.
%
% experiment - String identifier for data analysis experiment
%
% This function creates a parameter structure with input data specifications 
% as appropriate for the experiment. 
%
% The data is then generated and saved on disk using the circleData function. 
%
% Modified 2020/06/06

%% SET EXPERIMENT-SPECIFIC PARAMETERS
switch experiment

case 'a0.7'

    DataSpecs.Pars.f      = 1;     % frequency parameter
    DataSpecs.Pars.a      = 0.7;   % nonlinearity parameter
    DataSpecs.Pars.r1     = 1;     % ellipse axis 1
    DataSpecs.Pars.r2     = 1;     % ellipse axis 2
    DataSpecs.Time.dt     = 0.01 * sqrt( 2 );  % sampling interval
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nSSpin = 0;     % spinup samples
    DataSpecs.Time.nEL    = 0;     % embedding window length (extra samples)

otherwise
    
    error( 'Invalid experiment' )

end


%% SET PARAMETERS COMMON TO ALL EXPERIMENTS

% Extra samples before/after main time interval, total production samples
DataSpecs.Time.nXB    = 0;
DataSpecs.Time.nXA    = 0;
DataSpecs.Time.nS     = DataSpecs.Time.nSProd ...
                      + DataSpecs.Time.nEL ...
                      + DataSpecs.Time.nXB ...
                      + DataSpecs.Time.nXA; 

% Output options
DataSpecs.Opts.ifCenter = false;     % don't do data centering
DataSpecs.Opts.ifWrite  = true;       % write data to disk  

%% GENERATE DATA
circleData( DataSpecs );


