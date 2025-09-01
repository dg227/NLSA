function setDistanceNormalization( obj, rho, varargin )
% SETDISTANCENORMALIZATION  Set distance normalization data of an nlsaKernelDensity_ose_vb object
%
% Modified 2018/07/05

if ~isvector( rho ) || numel( rho ) ~= getNTotalSample( obj )
    error( 'Invalid normalization' )
end

if isrow( rho )
    rho = rho';
end

file = fullfile( getDensityPath( obj ), ... 
                 getDistanceNormalizationFile( obj ) );
save( file, 'rho', varargin{ : } )

