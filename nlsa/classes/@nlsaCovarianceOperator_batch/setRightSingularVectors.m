function setRightSingularVectors( obj, v, iB, varargin )
% SETRIGHTSINGULARVECTORS  Set right singular vector  data of an 
% nlsaCovarianceOperator object
%
% Modified 2014/07/16

partition = getPartition( obj );
if ~ischar( varargin{ 1 } )
    iR = varargin{ 1 }; % varargin{ 1 } stores realization
    varargin = varargin( 2 : end );
else
    [ iB, iR ]  = gl2loc( partition, iB );
end

nV = getNEigenfunction( obj );
if size( v, 1 ) ~= getBatchSize( partition( iR ), iB )
    error( 'Incompatible number of samples' )
end
if size( v, 2 ) ~= nV 
    error( 'Incompatible number of singular vectors' )
end

varNames = { 'v'  };

file = fullfile( getRightSingularVectorPath( obj ), ... 
                 getRightSingularVectorFile( obj, iB, iR ) );
save( file, varNames{ : }, varargin{ : } )


