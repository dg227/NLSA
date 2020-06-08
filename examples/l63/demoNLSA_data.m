function demoNLSA_data( experiment )
% DEMONLSA_DATA Helper function to generate datasets for NLSA demo for L63
% system.
%
% experiment - String identifier for data analysis experiment
%
% This function creates a parameter structure with input data specifications 
% as appropriate for the experiment. 
%
% The data is then generated and saved on disk using the l63Data function. 
%
% Modified 2020/06/06

%% SET EXPERIMENT-SPECIFIC PARAMETERS
switch experiment

% 6400 samples, sampling interval 0.01, no delay embedding 
case '6.4k_dt0.01_nEL0'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 0;     % embedding window length (extra samples)

% 6400 samples, sampling interval 0.01, 80 delays
case '6.4k_dt0.01_nEL80'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 80;     % embedding window length (extra samples)

% 6400 samples, sampling interval 0.01, 100 delays
case '6.4k_dt0.01_nEL100'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 100;   % embedding window length (extra samples)

% 6400 samples, sampling interval 0.01, 100 delays
case '6.4k_dt0.01_nEL150'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 150;   % embedding window length (extra samples)

% 6400 samples, sampling interval 0.01, 200 delays
case '6.4k_dt0.01_nEL200'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 200;   % embedding window length (extra samples)

% 6400 samples, sampling interval 0.01, 300 delays
case '6.4k_dt0.01_nEL300'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 300;   % embedding window length (extra samples)

% 6400 samples, sampling interval 0.01, 400 delays
case '6.4k_dt0.01_nEL400'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 6400;  % production samples
    DataSpecs.Time.nEL    = 400;   % embedding window length (extra samples)


% 64000 samples, sampling interval 0.01, no delay embedding 
case '64k_dt0.01_nEL0'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 64000; % production samples
    DataSpecs.Time.nEL    = 0;     % embedding window length (extra samples)

% 64000 samples, sampling interval 0.01, 400 delays
case '64k_dt0.01_nEL400'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 64000;  % production samples
    DataSpecs.Time.nEL    = 400;   % embedding window length (extra samples)

% 64000 samples, sampling interval 0.01, 800 delays
case '64k_dt0.01_nEL800'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 64000;  % production samples
    DataSpecs.Time.nEL    = 800;   % embedding window length (extra samples)

% 64000 samples, sampling interval 0.01, 400 delays
case '64k_dt0.01_nEL1600'

    DataSpecs.Time.dt     = 0.01;  % sampling interval
    DataSpecs.Time.nSSpin = 64000; % spinup samples
    DataSpecs.Time.nSProd = 64000;  % production samples
    DataSpecs.Time.nEL    = 1600;   % embedding window length (extra samples)

otherwise
    
    error( 'Invalid experiment' )

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
DataSpecs.Ode.x0     = [ 0 1 1.05 ]; % initial conditions
DataSpecs.Ode.relTol = 1E-8;         % relative tolerance for ODE solver 

% Output options
DataSpecs.Opts.ifCenter = false;     % don't do data centering
DataSpecs.Opts.ifWrite  = true;       % write data to disk  

%% GENERATE DATA
l63Data( DataSpecs );


