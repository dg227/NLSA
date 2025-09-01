function Data = l63Data(DataSpecs)
% ROSSLERDATA Integrate the Rossler model, and output potentially partial
% observations in format appropriate for NLSA code.
%
% DataSpecs is a data structure containing the specifications of the data to
% be generated.
%
% Data is a data structure containing the data read and associated attributes.
%
% DataSpecs has the following fields:
%
% Pars.a:            Rossler parameter a
% Pars.b:            Rossler parameter b
% Pars.c:            Rossler  parameter c
% Time.nS:           Number of production samples
% Time.nSSpin        Spinup samples
% Time.dt            Sampling interval
% Ode.x0:            Initial condition
% Ode.relTol         Relative tolerance for ODE solver
% Opts.idxX:         State vector components for partial obs
% Opts.ifCenter:     Data centering
% Opts.ifWrite:      Write data to disk
%
% idxX specifies the component indices for the partial obs.
% Output is written in an array x of dimension [nD nSProd], where
% nD = 3 in the case of full obs, nD = numel(idxX) in the case of partial
% obs, and nSProd is the number of "production" samples
%
% Modified 2023/10/15

%% UNPACK INPUT DATA STRUCTURE FOR CONVENIENCE
Pars   = DataSpecs.Pars;
Time   = DataSpecs.Time;
Ode    = DataSpecs.Ode;
Opts   = DataSpecs.Opts;

%% OUTPUT DIRECTORY
strDir = ['a'       num2str(Pars.a, '%1.3g') ...
          '_b'      num2str(Pars.b, '%1.3g') ...
          '_c'      num2str(Pars.c, '%1.3g') ...
          '_dt'     num2str(Time.dt, '%1.3g') ...
          '_x0'     sprintf('_%1.3g', Ode.x0) ...
          '_nS'     int2str(Time.nS) ...
          '_nSSpin' int2str(Time.nSSpin) ...
          '_relTol' num2str(Ode.relTol, '%1.3g') ...
          '_ifCent' int2str(Opts.ifCenter)];


%% INTEGRATE THE ROSSLER SYSTEM
% nS is the number of samples that will finally be retained
odeH = @(T, X) rossler(T, X, Pars.a, Pars.b, Pars.c );
t = (0 : Time.nS + Time.nSSpin - 1) * Time.dt;
[tOut, x] = ode45(odeH, t, Ode.x0, odeset('relTol', Ode.relTol, ...
                                          'absTol', eps));
x = x';
t = tOut';

t = t(Time.nSSpin + 1 : end) - t(Time.nSSpin + 1);
x = x(:, Time.nSSpin + 1 : end);

mu = mean(x, 2);
if Opts.ifCenter
    x  = x - mu;
end

%% WRITE DATA
if Opts.ifWrite
    pth = fullfile('./data', 'raw', strDir);
    if ~isdir(pth)
        mkdir(pth)
    end

    filenameOut = fullfile(pth, 'dataX.mat');
    save(filenameOut, '-v7.3', 'x', 'mu', 'Pars', 'Time', 'Ode')
end
Data.x = x;
Data.mu = mu;

%% CREATE DATASETS WITH PARTIAL OBS
if isfield(Opts, 'idxX')
    if ~iscell(Opts.idxX)
        Opts.idxX = {Opts.idxX};
    end
    for iObs = 1 : numel(Opts.idxX)
        x = Data.x(Opts.idxX{iObs}, :);
        filenameOut = ['dataX_idxX' sprintf('_%i', Opts.idxX{iObs})];
        filenameOut = fullfile(pth, [filenameOut '.mat']);
        save(filenameOut, '-v7.3', 'x')
    end
end
