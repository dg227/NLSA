function setDistanceNormalization( obj, rho, varargin )
% SETDISTANCENORMALIZATION  Set distance normalization data of an nlsaKernelDensity_vb object
%
% Modified 2015/12/16

if ~isvector( rho ) || numel( rho ) ~= getNTotalSample( obj )
    error( 'Invalid normalization' )
end

if isrow( rho )
    rho = rho';
end

file = fullfile( getDensityPath( obj ), ... 
                 getDistanceNormalizationFile( obj ) );
save( file, 'rho', varargin{ : } )

