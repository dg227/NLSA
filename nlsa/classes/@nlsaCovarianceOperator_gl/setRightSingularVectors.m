function setRightSingularVectors( obj, v, varargin )
% SETRIGHTSINGULARVECTORS  Set right singular vectors  of an 
% nlsaCovarianceOperator_gl object
%
% Modified 2014/07/16

if size( v, 1 ) ~= getNTotalSample( obj  )
    error( 'Incompatible number of samples' )
end
if size( v, 2 ) ~= getNEigenfunction( obj ) 
    error( 'Incompatible number of singular vectors' )
end
file = fullfile( getRightSingularVectorPath( obj ), ... 
                 getRightSingularVectorFile( obj ) );
save( file, 'v', varargin{ : } )

