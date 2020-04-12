function setEigenvalues( obj, gamma )
% SETEIGENVALUES  Save eigenvalues of an nlsaKoopmanOperator object
%
% Modified 2020/04/12

if ~isrow( gamma ) || ~isnumeric( gamma ) 
    error( 'Eigenvalues must be specified as a numeric row vector' )
end

if numel( gamma ) ~= getNEigenfunction( obj )
    error( 'Number of eigenvalues must be equal to number of eigenfunctions.' )end

save( fullfile( getEigenfunctionPath( obj ), getEigenvalueFile( obj ) ), ...
      'gamma' )

