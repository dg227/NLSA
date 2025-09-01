function setDensity( obj, q, varargin )
% SETDENSITY  Set density data of an nlsaKernelDensity_ose_fb object
%
% Modified 2018/07/08

if ~isvector( q ) || numel( q ) ~= getNTotalSample( obj )
    error( 'Invalid density' )
end

if isrow( q )
    q = q';
end

file = fullfile( getDensityPath( obj ), ... 
                 getDensityFile( obj ) );
save( file, 'q', varargin{ : } )

