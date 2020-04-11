function setEigenfunctionCoefficients( obj, c, varargin )
% SETEIGENFUNCTIONCOEFFICIENTS Set eigenfunction coefficients for an 
% nlsaKoopmanOperator object
%
% Modified 2020/04/11

if ~isnumeric( c ) || ~ismatrix( c ) 
    msgStr = [ 'Eigenfunction coefficients must be specified as a ' ...
               'numeric matrix.' ];
    error( msgStr ) 
end
if any( size( c ) ~= getNEigenfunction( obj ) )
    error( 'Incompatible number of eigenfunction coefficients' )
end

file = fullfile( getEigenfunctionPath( obj ), ... 
                 getEigenfunctionCoeffientFile( obj ) );
save( file, 'c', varargin{ : } )

