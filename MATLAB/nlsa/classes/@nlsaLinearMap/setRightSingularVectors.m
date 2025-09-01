function setRightSingularVectors( obj, v, varargin )
% SETRIGHTSINGULARVECTORS  Set right singular vectors  of an 
% nlsaLinearMap_gl object
%
% Modified 2015/10/19

if ~all( size( v ) == getNEigenfunction( obj ) * [ 1 1 ] )
    error( 'Incompatible number of singular vectors' )
end
file = fullfile( getRightSingularVectorPath( obj ), ... 
                 getRightSingularVectorFile( obj ) );
save( file, 'v', varargin{ : } )

