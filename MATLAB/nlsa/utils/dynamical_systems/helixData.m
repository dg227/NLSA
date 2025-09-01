function [ Data, DataSpecs ] = helixData( DataSpecs )
% HELIXDATA Nonstationary data with trend and cyclical components 
%
% The generated trejectory is embedded in R3. The trend function is 
% 
% h(t) = a*t + b*(t - c)^2 
%
% DataSpecs is a data structure containing the specifications of the data to
% be generated.
%
% Data is a data structure containing the data read and associated attributes.
%
% DataSpecs has the following fields:
%
% Pars.a:            Linear trend coefficient
% Pars.b:            Quadratic trend coefficient 
% Pars.c:            Quadratic offset 
% Time.dt:           Sampling interval
% Time.nS:           Number of samples
% Time.dt            Sampling interval
% Opts.idxX:         State vector components for partial obs
% Opts.ifWrite:      Write data to disk 
%
% Modified 2021/01/20


%% UNPACK INPUT DATA STRUCTURE FOR CONVENIENCE
Pars   = DataSpecs.Pars;
Time   = DataSpecs.Time;
Opts   = DataSpecs.Opts;

%% OUTPUT DIRECTORY
strDir = [ 'a'       num2str( Pars.a, '%1.2f' ) ...
           '_b'      num2str( Pars.b, '%1.2f' ) ...
           '_c'      num2str( Pars.c, '%1.2f' ) ...
           '_dt'     num2str( Time.dt ) ...
           '_nS'     int2str( Time.nS ) ];

%% GENERATE TRAJECTORY
lag    = round( pi / 2 / Time.dt );
a      = Pars.a;
b      = Pars.b;
c      = Pars.c;
nSTot  = Time.nS + 2 * lag;           % total number of samples 
t      = ( 0 : nSTot - 1 ) * Time.dt; % timestamps  
g      = cos( t ) + a * t + b * ( t - c ) .^ 2;

%% EMBED INTO DATA SPACE
x   = [ g( 1       : end - 2 * lag ); ...
        g( 1 + lag : end - lag ); ...
        g( 1 + 2 * lag : end ) ];

%% WRITE DATA
if Opts.ifWrite
    pth = fullfile( './data', 'raw', strDir );
    if ~isdir( pth )
        mkdir( pth )
    end
    
    filenameOut = fullfile( pth, 'dataX.mat' );
    save( filenameOut, '-v7.3', 'x', 'g', 'Pars', 'Time' ) 
end
Data.x = x;


%% CREATE DATASETS WITH PARTIAL OBS
if isfield( Opts, 'idxX' )
    for iObs = 1 : numel( Opts.idxX )
        x = Data.x( Opts.idxX{ iObs }, : );
        filenameOut = [ 'dataX_idxX' sprintf( '_%i', Opts.idxX{ iObs } ) ];
        filenameOut = fullfile( pth, [ filenameOut '.mat' ] );  
        save( filenameOut, '-v7.3', 'x' )
    end
end
