function setSpectralEntropy( obj, s )
% SETSPECTRALENTROPY  Save spectral entropy values of an 
% nlsaCovarianceOperator object
%
% Modified 2014/07/16

if ~isvector( s ) || numel( s ) ~= getNEigenfunction( obj ) - 1
    error( 'Incompatible number of spectral entropy values' )
end
save( fullfile( getSingularValuePath( obj ), getSpectralEntropyFile( obj ) ), ...
      's' )
