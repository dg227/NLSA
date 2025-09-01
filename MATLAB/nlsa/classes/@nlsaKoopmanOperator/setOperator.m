function setOperator( obj, V, varargin )
% SETOPERATOR  Set operator data of an nlsaKoopmanOperator object 
%
% Modified 2020/04/10

nPhi = getNEigenfunction( obj );

if ~isnumeric( V ) || ~ismatrix( V ) || any( size( V ) ~= [ nPhi nPhi ] )
    msgStr = [ 'Operator data must be passed as a numeric matrix of ' ...
               'dimension equal to the number of basis functions.' ];
    error( msgStr )
end

file = fullfile( getOperatorPath( obj ), getOperatorFile( obj ) );
varNames = { 'V' };
save( file, varNames{ : }, varargin{ : } )

