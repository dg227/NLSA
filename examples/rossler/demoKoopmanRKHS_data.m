function demoKoopmanRKHS_data(experiment)
% DEMOKOOPMANRKHS_DATA Helper function to generate datasets for Koopman RKHS
% demo for Rossler system.
%
% experiment - String identifier for data analysis experiment
%
% This function creates a parameter structure with input data specifications
% as appropriate for the experiment.
%
% The data is then generated and saved on disk using the l63Data function.
%
% Modified 2023/10/15

%% SET EXPERIMENT-SPECIFIC PARAMETERS
switch experiment

% 6400 samples, sampling interval 0.04, no delay embedding
case '640_dt0.04_nEL0'

    DataSpecs.Time.dt     = 0.04;  % sampling interval
    DataSpecs.Time.nSSpin = 640;  % spinup samples
    DataSpecs.Time.nSProd = 640;  % production samples
    DataSpecs.Time.nEL    = 0;     % embedding window length (extra samples)

% 6400 samples, sampling interval 0.04, no delay embedding
case '6.4k_dt0.04_nEL0'

    DataSpecs.Time.dt     = 0.04;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 0;     % embedding window length (extra samples)

% 6400 samples, sampling interval 0.04, 400 delays
case '6.4k_dt0.04_nEL800'

    DataSpecs.Time.dt     = 0.04;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 800;   % embedding window length (extra samples)

% 64000 samples, sampling interval 0.04, no delay embedding
case '64k_dt0.04_nEL0'

    DataSpecs.Time.dt     = 0.04;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 64000; % production samples
    DataSpecs.Time.nEL    = 0;     % embedding window length (extra samples)

% 64000 samples, sampling interval 0.04, 800 delays
case '64k_dt0.04_nEL800'

    DataSpecs.Time.dt     = 0.04;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 64000;  % production samples
    DataSpecs.Time.nEL    = 800;   % embedding window length (extra samples)

otherwise

    error('Invalid experiment')

end


%% SET PARAMETERS COMMON TO ALL EXPERIMENTS

% Standard Rossler parameters
DataSpecs.Pars.a  = 0.1;         % Rossler parameter a
DataSpecs.Pars.b  = 0.1;         % Rossler parameter b
DataSpecs.Pars.c  = 14;          % Rossler parameter c

% Extra samples before/after main time interval, total production samples
DataSpecs.Time.nXB    = 0;
DataSpecs.Time.nXA    = 0;
DataSpecs.Time.nS     = DataSpecs.Time.nSProd ...
                      + DataSpecs.Time.nEL ...
                      + DataSpecs.Time.nXB ...
                      + DataSpecs.Time.nXA;

% Initial conditions and tolerance for ODE solver
DataSpecs.Ode.x0     = [0 1 1.05]; % initial conditions
DataSpecs.Ode.relTol = 1E-8;         % relative tolerance for ODE solver

% Output options
DataSpecs.Opts.ifCenter = false;     % don't do data centering
DataSpecs.Opts.ifWrite  = true;       % write data to disk

%% GENERATE DATA
rosslerData(DataSpecs);
