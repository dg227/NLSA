function setLinearMap( obj, a, varargin )
% SETLINEARMAP  Set linear map matrix elements of an nlsaCovarianceOperator_gl 
% object
%
% Modified 2015/10/19


if ~ismatrix( a ) || ~all( size( a )  == [ getTotalDimension( obj ) ...
                                           getNEigenfunction( obj  ) ] ) 
    error( 'Incompatible linear map array size' )
end

file = fullfile( getOperatorPath( obj ), ... 
                 getLinearMapFile( obj ) );
save( file, 'a', varargin{ : } )

