function setLeftEigenfunctionCoefficients( obj, c, varargin )
% SETLEFTEIGENFUNCTIONCOEFFICIENTS Set left eigenfunction coefficients for an 
% nlsaKoopmanOperator object
%
% Modified 2020/08/27

if ~isnumeric( c ) || ~ismatrix( c ) 
    msgStr = [ 'Eigenfunction coefficients must be specified as a ' ...
               'numeric matrix.' ];
    error( msgStr ) 
end
if size( c, 1 ) ~= numel( getBasisFunctionIndices( obj ) ) ...
   || size( c, 2 ) ~= getNEigenfunction( obj ) 
    error( 'Incompatible number of eigenfunction coefficients' )
end

file = fullfile( getEigenfunctionPath( obj ), ... 
                 getEigenfunctionCoefficientFile( obj ) );
save( file, 'c', varargin{ : } )

