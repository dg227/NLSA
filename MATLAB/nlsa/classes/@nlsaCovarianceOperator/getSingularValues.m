function s = getSingularValues( obj )
% GETSINGULARVALUES  Read singular values of an nlsaCovarianceOperator object
%
% Modified 2014/07/16

load( fullfile( getSingularValuePath( obj ), getSingularValueFile( obj ) ), ...
      's' )
