function obj = setDimension( obj, nD )
% SETDIMENSION  Set dimension of nlsaCovarianceOperator objects
%
% Modified 2014/08/07

for iC = 2 : numel( nD )
    nD( iC ) = nD( iC - 1 ) + nD( iC );
end
obj = setSpatialPartition( obj, nlsaPartition( 'idx', nD ) );
