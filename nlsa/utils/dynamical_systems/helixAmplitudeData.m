function [ Data, DataSpecs ] = helixAmplitudeData( DataSpecs )
% HELIXAMMPLITUDEDATA Nonstationary data with trend and cyclical components 
%
% The generated trejectory is embedded in R3. The trend function is 
% 
% h(t) = (1 + a * sin( t / T * pi ) + b * t.
%
% DataSpecs is a data structure containing the specifications of the data to
% be generated.
%
% Data is a data structure containing the data read and associated attributes.
%
% DataSpecs has the following fields:
%
% Pars.a:            Amplitude coefficient 
% Pars.T:            Trend period parameter
% Time.dt:           Sampling interval
% Opts.idxX:         State vector components for partial obs
% Opts.ifWrite:      Write data to disk 
%
% Modified 2022/02/02


%% UNPACK INPUT DATA STRUCTURE FOR CONVENIENCE
Pars   = DataSpecs.Pars;
Time   = DataSpecs.Time;
Opts   = DataSpecs.Opts;

%% OUTPUT DIRECTORY
strDir = [ 'a'      num2str( Pars.a, '%1.2f' ) ...
           '_b'     num2str( Pars.b, '%1.2f' ) ...
           '_T'     num2str( Pars.T, '%1.2f' ) ...
           '_dt'    num2str( Time.dt ) ];

%% GENERATE TRAJECTORY
lag    = round( pi / 2 / Time.dt );
t      = 0 : Time.dt : Pars.T; % timestamps  
g      = ( 1 + Pars.a * sin( t / Pars.T * pi ) + Pars.b * t ) .* cos( t );

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
