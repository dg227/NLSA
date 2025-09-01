function demoKAF_test_test_data( experiment )
% DEMOKAF_TEST_DATA Helper function to generate test (forecast verification) 
% datasets for KAF demo for L63 system.
%
% experiment - String identifier for data analysis experiment
%
% This function creates a parameter structure with input data specifications 
% as appropriate for the experiment. 
%
% The data is then generated and saved on disk using the l63Data function. 
%
% Modified 2021/02/27

%% SET EXPERIMENT-SPECIFIC PARAMETERS
switch experiment

% 6400 samples, sampling interval 0.01, no delay embedding, full obs. 
case '6.4k_dt0.01_idxX1_2_3_nEL0'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 0;     % embedding window length (extra samples)

    DataSpecs.Ode.x0     = [ 0.2 1.2 1.25 ]; % initial conditions

% 6400 samples, sampling interval 0.01, no delay embedding, x1 only 
case '6.4k_dt0.01_idxX1_nEL0'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 0;     % embedding window length (extra samples)

    DataSpecs.Ode.x0     = [ 0.2 1.2 1.25 ]; % initial conditions

    DataSpecs.Opts.idxX = 1; % partial observations

% 6400 samples, sampling interval 0.01, no delay embedding, full obs. 
case '6.4k_dt0.01_idxX1_2_3_nEL15'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 15;     % embedding window length (extra samples)

    DataSpecs.Ode.x0     = [ 0.2 1.2 1.25 ]; % initial conditions

% 6400 samples, sampling interval 0.01, 15 delays, x1 only 
case '6.4k_dt0.01_idxX1_nEL15'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 15;    % embedding window length (extra samples)

    DataSpecs.Ode.x0     = [ 0.2 1.2 1.25 ]; % initial conditions

    DataSpecs.Opts.idxX = 1; % partial observations


otherwise
    
    error( 'Invalid experiment' )

end


%% SET PARAMETERS COMMON TO ALL EXPERIMENTS
% Standard L63 parameters
DataSpecs.Pars.beta   = 8/3;         % L63 parameter beta
DataSpecs.Pars.rho    = 28;          % L63 parameter rho
DataSpecs.Pars.sigma  = 10;          % L63 parameter sigma

% Extra samples before/after main time interval, total production samples.
% We add samples after the main interval to provide a complete set of %
% training samples for forecasting.
DataSpecs.Time.nXB    = 0;
DataSpecs.Time.nXA    = 500;
DataSpecs.Time.nS     = DataSpecs.Time.nSProd ...
                      + DataSpecs.Time.nEL ...
                      + DataSpecs.Time.nXB ...
                      + DataSpecs.Time.nXA; 

% Tolerance for ODE solver
DataSpecs.Ode.relTol = 1E-8;             % relative tolerance for ODE solver 

% Output options
DataSpecs.Opts.ifCenter = false;     % don't do data centering
DataSpecs.Opts.ifWrite  = true;      % write data to disk  

%% GENERATE DATA
l63Data( DataSpecs );


