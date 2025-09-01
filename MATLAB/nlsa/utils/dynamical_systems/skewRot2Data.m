function Data = skewRot2Data( DataSpecs )
% SKEWROT2DATA Integrate the skew-rotation model, and output potentially 
% partial observations in format appropriate for NLSA code.
%
% DataSpecs is a data structure containing the specifications of the data to
% be generated.
%
% Data is a data structure containing the data read and associated attributes.
%
% DataSpecs has the following fields:
% 
% Pars.a1:         Frequency parameter along first coordinate (base)
% Pars.a2          Frequency parameter along second coordinate (base)
% Pars.a3:         Frequency parameter along third coordinate (fiber)
% Pars.b1:         Coupling parameter of fiber dynamics with first coordinate
% Pars.b2:         Coupling parameter of fiber dynamics with second coordinate
% Pars.k:          Bump function variance parameter (for 'r1' embedding)
% Pars.obsMap:     String identifier for observation map ('r1' or 'r6')
% Time.nS:         Number of production samples
% Time.nSSpin      Spinup samples
% Time.dt          Sampling interval
% Ode.x0:          Initial condition
% Ode.relTol       Relative tolerance for ODE solver
% Opts.idxX:       State vector components for partial obs
% Opts.ifCenter:   Data centering
% Opts.ifWrite:    Write data to disk 
%
% idxX specifies the component indices for the partial obs.
% Output is written in an array x of dimension [ nD nSProd ], where 
% nD = 6 in the case of full obs, nD = numel( idxX ) in the case of partial
% obs, and nSProd is the number of "production" samples
%
% Modified 2022/02/02

%% UNPACK INPUT DATA STRUCTURE FOR CONVENIENCE
Pars   = DataSpecs.Pars;
Time   = DataSpecs.Time;
Ode    = DataSpecs.Ode; 
Opts   = DataSpecs.Opts;

%% OUTPUT DIRECTORY
strDir = [ 'a1'      num2str( Pars.a1, '%1.3g' ) ...
           '_a2'     num2str( Pars.a2, '%1.3g' ) ...
           '_a3'     num2str( Pars.a3, '%1.3g' ) ...
           '_b1'     num2str( Pars.b1, '%1.3g' ) ...
           '_b2'     num2str( Pars.b2, '%1.3g' ) ...
           '_dt'     num2str( Time.dt, '%1.3g' ) ...
           '_nS'     int2str( Time.nS ) ...
           '_nSSpin' int2str( Time.nSSpin ) ...
           '_relTol' num2str( Ode.relTol, '%1.3g' ) ];

switch Pars.obsMap
    case 'r1' 
        strDirObs = [ 'r1' ...
                      '_kappa' num2str( Pars.k, '%1.3g' ) ... 
                      '_ifCent'  int2str( Opts.ifCenter ) ];
    case 'r6' % flat embedding in R^6
        strDirObs = [ 'r6' ...
                      '_ifCent'  int2str( Opts.ifCenter ) ];
end

%% INTEGRATE THE SKEW-ROTATION SYSTEM
% nS is the number of samples that will finally be retained 
odeH = @( T, X ) skewRot2( T, X, Pars.a1, Pars.a2, Pars.a3 );
t = ( 0 : Time.nS + Time.nSSpin - 1 ) * Time.dt;
theta0 = [ 0 0 0 ];
[ tOut, theta ] = ode45( odeH, t, theta0, odeset( 'relTol', Ode.relTol, ...
                                                  'absTol', eps ) );
theta = theta';
t = tOut';

t     = t( Time.nSSpin + 1 : end ) - t( Time.nSSpin + 1 );
theta = theta( :, Time.nSSpin + 1 : end );


%% EMBED INTO DATA SPACE
% x is an array of size [ nD nS ] where nD is the dimension of data space
switch Pars.obsMap
    case 'r1'
        x = exp( Pars.k * cos( theta( 3, : ) ) ) / besseli( 0, Pars.k );
    case 'r6'
        x = [ cos( theta( 1, : ) ) ...
              sin( theta( 1, : ) ) ...
              cos( theta( 2, : ) ) ...
              sin( theta( 2, : ) ) ...
              cos( theta( 3, : ) ) ...
              sin( theta( 3, : ) ) ];
end

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
    save( filenameOut, '-v7.3', 'x', 'mu', 'Pars', 'Time', 'Ode' ) 
end
Data.x = x;
Data.mu = mu;

%% CREATE DATASETS WITH PARTIAL OBS
if isfield( Opts, 'idxX' )
    if ~iscell( Opts.idxX )
        Opts.idxX = { Opts.idxX };
    end
    for iObs = 1 : numel( Opts.idxX )
        x = Data.x( Opts.idxX{ iObs }, : );
        filenameOut = [ 'dataX_idxX' sprintf( '_%i', Opts.idxX{ iObs } ) ];
        filenameOut = fullfile( pth, [ filenameOut '.mat' ] );  
        save( filenameOut, '-v7.3', 'x' )
    end
end

