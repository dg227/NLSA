function setDoubleSum( obj, dSum, varargin )
% SETDOUBLESUM  Set double sum data of an nlsaKernelDensity_fb object
%
% Modified 2015/04/06

if ~isvector( dSum ) || numel( dSum ) ~= getNBandwidth( obj )
    error( 'Invalid double sum' )
end

if iscolumn( dSum )
    dSum = dSum';
end

file = fullfile( getDensityPath( obj ), ... 
                 getDoubleSumFile( obj ) );
save( file, 'dSum', varargin{ : } )

