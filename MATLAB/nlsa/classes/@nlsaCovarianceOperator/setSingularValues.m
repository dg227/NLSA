function setSingularValues( obj, s )
% SETSINGULARVALUES  Save singular values of an nlsaCovarianceOperator object
%
% Modified 2015/10/19

if ~isvector( s ) || numel( s ) ~= min( getNEigenfunction( obj ), ...
                                        sum( getDimension( obj ) ) )
    error( 'Incompatible number of singular values' )
end
save( fullfile( getSingularValuePath( obj ), getSingularValueFile( obj ) ), ...
      's' )
