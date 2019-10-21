function setDoubleSum( obj, dSum, varargin )
% SETDOUBLESUM  Set double sum data of an nlsaDiffusionOperator_gl_mb object
%
% Modified 2015/05/08

if ~isvector( dSum ) || numel( dSum ) ~= getNBandwidth( obj )
    error( 'Invalid double sum' )
end

if iscolumn( dSum )
    dSum = dSum';
end

file = fullfile( getOperatorPath( obj ), ... 
                 getDoubleSumFile( obj ) );
save( file, 'dSum', varargin{ : } )

