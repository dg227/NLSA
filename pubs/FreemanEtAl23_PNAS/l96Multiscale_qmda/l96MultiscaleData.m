function Data = l96MultiscaleData(DataSpecs)
% L96MULTISCALEDATA Integrate the 2-level Lorentz 96 model, and output 
% potentially partial observations in format appropriate for NLSA code.
%
% DataSpecs is a data structure containing the specifications of the data to
% be generated.
%
% Data is a data structure containing the data read and associated attributes.
%
% DataSpecs has the following fields:
% 
% Pars.nX:       Number of slow variables
% Pars.nY:       Number of fast variables
% Pars.F:        Forcing parameter 
% Pars.hX:       Coupling parameter from slow to fast variables
% Pars.hY:       Coupling parameter from fast to slow variables 
% Pars.epsilon:  Timescale parameter for fast variables
% Time.nS:       Number of production samples
% Time.nSSpin    Spinup samples
% Time.dt        Sampling interval
% Ode.x0:        Initial condition for slow variables
% Ode.y0:        Initial condition for fast variables
% Ode.relTol     Relative tolerance for ODE solver
% Ode.absTol     Absolute tolerance for ODE solver
% Opts.idxX:     State vector components for partial obs (slow variables)
% Opts.idxY:     State vector components for partial obs (fast variables)
% Opts.ifCenter: Data centering
% Opts.ifWrite:  Write data to disk 
%
% Output is written in a arrays x and y of dimension [nDX nSProd] and 
% [nDY nSProd], respectively, where nDX is equal to the number of elements of
% Pars.idxX and nDY is equal to the number of elements of Pars.idxY.
%
% The array y is not output if Opts.idxY is not specified.
%
% If Opts.ifWrite is set to true, the output is written in a .mat file in
% a directory name generated from the model and integration parameters/
%
% WARNING: The directory name depends only on the first components of 
% Ode.x0 and Ode.y0. Initial condition vectors tht differ in the subsequent
% vector components will lead to data overwrites.
%
% Modified 2021/07/15

%% UNPACK INPUT DATA STRUCTURE FOR CONVENIENCE
Pars   = DataSpecs.Pars;
Time   = DataSpecs.Time;
Ode    = DataSpecs.Ode; 
Opts   = DataSpecs.Opts;


%% OUTPUT DIRECTORY
strDir = ['nX'      int2str(Pars.nX) ...
          '_nY'     int2str(Pars.nY) ...
          '_F'      num2str(Pars.F, '%1.3g') ...
          '_hX'     num2str(Pars.hX, '%1.3g') ...
          '_hY'     num2str(Pars.hY, '%1.3g') ...
          '_eps'    num2str(Pars.epsilon, '%1.3g') ...
          '_dt'     num2str(Time.dt, '%1.3g') ...
          '_x0'     num2str(Ode.x0(1), '%1.3g') ...
          '_y0'     num2str(Ode.y0(1), '%1.3g') ...
          '_nS'     int2str(Time.nS) ...
          '_nSSpin' int2str(Time.nSSpin) ...
          '_absTol' num2str(Ode.absTol, '%1.3g') ...
          '_relTol' num2str(Ode.relTol, '%1.3g') ...
          '_ifCent' int2str(Opts.ifCenter)];


%% INTEGRATE THE 2-LEVEL L96 SYSTEM
% nS is the number of samples that will finally be retained 
odeH = @(T, X) l96Multiscale(T, X, Pars.F, Pars.hX, Pars.hY, ...
                                      Pars.epsilon, Pars.nX, Pars.nY);
t = (0 : Time.nS + Time.nSSpin - 1) * Time.dt;
if isscalar(Ode.x0)
    x0 = zeros(1, Pars.nX);
    x0(1) = Ode.x0;
else 
    x0 = Ode.x0;
end
if isscalar(Ode.y0)
    y0 = zeros(1, Pars.nY);
    y0(1) = Ode.y0;
    y0 = repmat(y0, 1, Pars.nX);
else 
    y0 = Ode.y0;
end
z0 = [x0 y0];
[tOut, z] = ode15s(odeH, t, z0, odeset('relTol', Ode.relTol, ...
                                           'absTol', Ode.absTol));
x = z(Time.nSSpin + 1 : end, 1 : Pars.nX)';
y = z(Time.nSSpin + 1 : end, Pars.nX + 1 : end)';
t = tOut';
t = t(Time.nSSpin + 1 : end) - t(Time.nSSpin + 1);

muX = mean(x, 2);
muY = mean(y, 2);
if Opts.ifCenter
    x = x - muX;
    y = y - muY;
end


%% WRITE DATA
if Opts.ifWrite
    pth = fullfile('./data', 'raw', strDir);
    if ~isdir(pth)
        mkdir(pth)
    end
    
    filenameOut = fullfile(pth, 'dataX.mat');
    save(filenameOut, '-v7.3', 'x', 'y', 'muX', 'muY', 'Pars', 'Time', 'Ode') 
end
Data.x = x;
Data.muX = muX;
Data.y = y;
Data.muY = muY;


%% CREATE DATASETS WITH PARTIAL OBSERVATIONS
if isfield(Opts, 'idxX')
    if ~iscell(Opts.idxX)
        Opts.idxX = { Opts.idxX };
    end
    for iObs = 1 : numel(Opts.idxX)
        x = Data.x(Opts.idxX{ iObs }, :);
        filenameOut = ['dataX_' idx2str(Opts.idxX{ iObs }, 'idxX')];
        filenameOut = fullfile(pth, [filenameOut '.mat']);  
        save(filenameOut, '-v7.3', 'x')
    end
end
if isfield(Opts, 'idxY')
    if ~iscell(Opts.idxY)
        Opts.idxY = { Opts.idxY };
    end
    for iObs = 1 : numel(Opts.idxY)
        y = Data.y(Opts.idxY{ iObs }, :);
        filenameOut = ['dataX_' idx2str(Opts.idxY{ iObs }, 'idxY')];
        filenameOut = fullfile(pth, [filenameOut '.mat']);  
        save(filenameOut, '-v7.3', 'y')
    end
end
