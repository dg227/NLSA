function [ Data, DataSpecs ] = circleData( DataSpecs )
% CIRCLEDATA Integrate variable speed dynamical system on the circle:
%
% dtheta/dt = f * ( 1 - sqrt( 1 - a ) * sin( theta ) )
% dtheta/dt = f * ( 1 + sqrt( 1 - a ) * sin( theta ) )
% 
% theta( 0 ) = 0,
%
% where f is a frequency parameter, and a controls the nonlinerity of the 
% flow. The parameter value a = 1 correspond to linear (fixed-speed) flow on 
% the circle.
%
% The generated trejectory is embedded as an ellipse in R2,
%
% [ x1, x2  ] = [ r1 * cos( theta ), r2 * sin( theta ) ].
%
% DataSpecs is a data structure containing the specifications of the data to
% be generated.
%
% Data is a data structure containing the data read and associated attributes.
%
% DataSpecs has the following fields:
%
% Pars.f:            Frequency parameter
% Pars.a:            Nonlinearity parameter
% Pars.r1            Ellipse axis 1
% Pars.r2            Ellipse axis 2
% Time.nS:           Number of production samples
% Time.nSSpin        Spinup samples
% Time.dt            Sampling interval
% Opts.idxX:         State vector components for partial obs
% Opts.ifCenter:     Data centering
% Opts.ifWrite:      Write data to disk 
%
% Modified 2020/07/16


%% UNPACK INPUT DATA STRUCTURE FOR CONVENIENCE
Pars   = DataSpecs.Pars;
Time   = DataSpecs.Time;
Opts   = DataSpecs.Opts;

%% OUTPUT DIRECTORY
strDir = [ 'f'       num2str( Pars.f, '%1.2f' ) ...
           '_a'      num2str( Pars.a, '%1.2f' ) ...
           '_dt'     num2str( Time.dt ) ...
           '_nS'     int2str( Time.nS ) ...
           '_nSSpin' int2str( Time.nSSpin ) ...
           '_r1'     num2str( Pars.r1, '%1.2f' ) ...
           '_r2'     num2str( Pars.r2, '%1.2f' ) ...
           '_ifCent' int2str( Opts.ifCenter ) ];

%% GENERATE TRAJECTORY
a      = Pars.a;
f      = Pars.f;
nSTot  = Time.nS + Time.nSSpin; % total number of samples 
t      = ( 0 : nSTot - 1 ) * Time.dt; % timestamps  
if a ~= 1
    theta = 2 * acot( - ( sqrt( 1 - a ) ) ...
                      + sqrt( a ) * cot( sqrt( a ) * f * t / 2 ) ); 
else
    theta = f * t + pi / 2;
end
t = t( Time.nSSpin + 1 : end );
theta = theta( Time.nSSpin + 1 : end );

%% EMBED INTO DATA SPACE
x = zeros( 2, Time.nS );
x( 1, : ) = Pars.r1 * cos( theta );
x( 2, : ) = Pars.r2 * sin( theta );

% Mean and data centering
mu = mean( x, 2 );
if Opts.ifCenter
    x  = x - mu;
end

%% WRITE DATA
if Opts.ifWrite
    pth = fullfile( './data', 'raw', strDir );
    if ~isdir( pth )
        mkdir( pth )
    end
    
    filenameOut = fullfile( pth, 'dataX.mat' );
    save( filenameOut, '-v7.3', 'x', 'mu', 'Pars', 'Time' ) 
end
Data.x = x;
Data.mu = mu;

%% CREATE DATASETS WITH PARTIAL OBS
if isfield( Opts, 'idxX' )
    for iObs = 1 : numel( idxX )
        x = Data.x( Opts.idxX{ iObs }, : );
        filenameOut = [ 'dataX_idxX' sprintf( '_%i', Opts.idxX{ iObs } ) ];
        filenameOut = fullfile( pth, [ filenameOut '.mat' ] );  
        save( filenameOut, '-v7.3', 'x' )
    end
end
