function setEigenvalues( obj, lambda )
% SETEIGENVALUES  Save eigenvalues of an nlsaDiffusionOperator object
%
% Modified 2014/12/15

if ~isvector( lambda ) || numel( lambda ) ~= getNEigenfunction( obj )
    error( 'Incompatible number of eigenvalues' )
end
save( fullfile( getEigenfunctionPath( obj ), getEigenvalueFile( obj ) ), ...
      'lambda' )
