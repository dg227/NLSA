function demoKoopman_data(experiment)
% DEMOKOOPMAN_DATA Helper function to generate datasets for Koopman demo for 
% L63 system.
%
% experiment - String identifier for data analysis experiment
%
% This function creates a parameter structure with input data specifications 
% as appropriate for the experiment. 
%
% The data is then generated and saved on disk using the l63Data function. 
%
% Modified 2021/09/13

%% DEFAULT CASE
% 16000 samples, sampling interval 0.01, 800 delays
if nargin == 0
    experiment = '16k_dt0.01_nEL800';
end
        
%% SET EXPERIMENT-SPECIFIC PARAMETERS
switch experiment

% 16000 samples, sampling interval 0.01, 800 delays
case '16k_dt0.01_nEL800'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 16000; % production samples
    DataSpecs.Time.nEL    = 800;   % embedding window length (extra samples)

otherwise
    error('Invalid experiment')
end


%% SET PARAMETERS COMMON TO ALL EXPERIMENTS

% Standard L63 parameters
DataSpecs.Pars.beta   = 8/3;         % L63 parameter beta
DataSpecs.Pars.rho    = 28;          % L63 parameter rho
DataSpecs.Pars.sigma  = 10;          % L63 parameter sigma

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
l63Data(DataSpecs);


