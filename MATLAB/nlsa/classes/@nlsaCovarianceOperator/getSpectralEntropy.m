function s = getSpectralEntropy( obj )
% GETSPECTRALENTROPY  Read spectral entropy of an nlsaCovarianceOperator object
%
% Modified 2014/07/16

load( fullfile( getSingularValuePath( obj ), getSpectralEntropyFile( obj ) ), ...
      'ent' )
