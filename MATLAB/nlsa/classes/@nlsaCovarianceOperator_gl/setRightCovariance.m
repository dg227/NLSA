function setRightCovariance( obj, cv, varargin )
% SETRIGHTCOVARIANCE  Set right covariance matrix of an 
% nlsaCovarianceOperator_gl object
%
% Modified 2016/06/03

if ~ismatrix( cv ) || any( size( cv ) ~= getNTotalSample( obj  ) )
    error( 'Incompatible covariance array size' )
end

file = fullfile( getOperatorPath( obj ), ... 
                 getRightCovarianceFile( obj ) );
save( file, 'cv', varargin{ : } )

