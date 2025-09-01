function lambda = getEigenvalues( obj )
% GETEIGENVALUES  Read eigenvalues of an nlsaDiffusionOperator object
%
% Modified 2014/04/08

load( fullfile( getEigenfunctionPath( obj ), getEigenvalueFile( obj ) ), ...
      'lambda' )
