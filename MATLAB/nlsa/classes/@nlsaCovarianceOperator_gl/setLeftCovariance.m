function setLeftCovariance( obj, cu, varargin )
% SETLEFTCOVARIANCE  Set left covariance matrix of an 
% nlsaCovarianceOperator_gl object
%
% Modified 2014/07/16

if ~ismatrix( u ) || any( size( u ) ) ~= getNSample( getSpatialPartition( obj )  )
    error( 'Incompatible covariance array size' )
end

file = fullfile( getOperatorPath( obj ), ... 
                 getLeftCovarianceFile( obj ) );
save( file, 'cu', varargin{ : } )

