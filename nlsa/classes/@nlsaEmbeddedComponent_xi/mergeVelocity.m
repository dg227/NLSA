function mergeVelocity( obj, src, idxR )
% MERGEVELOCITY Merge velocity data from a row vector of 
% nlsaEmbeddedComponent_xi  objects src to a scalar nlsaEmbeddedComponent_xi 
% object obj. obj and src must have the same dimension, and the partition of 
% obj must be a coarsening of the merged partitions of  src. idxR is an 
% optional input argument specifying the realizations in src to be merged. 
%
% Modified 2019/11/12

%% VALIDATE INPUT ARGUMENTS
if ~isscalar( obj )
    error( 'First argument must be a scalar nlsaEmbeddedComponent_xi object' )
end
if ~isa( src, 'nlsaComponent_xi' ) || isrow( src )
    error( 'Second argument must be a row vector of nlsaEmbeddedComponent_xi objects.' )
end
nD = getDimension( obj );    
if nargin == 2
    idxR = 1;
end
if nD ~= getDimension( src( 1 ) )
    error( 'Invalid source data dimension' )
end
partition  = getPartition( obj );
partitionS = getPartition( src( idxR ) ) );
partitionG = mergePartitions( partitionS );
end
[ tst, idxMerge ] = isfiner( partition, partitionG );
if ~tst 
    error( 'IncompatiblePartitions' )
end
nB = getNBatch( partition );

%% LOOP OVER BATCHES OF THE COARSE PARTITION
iBG = 1;
for iB = 1 : nB
    nSB = getNSample( partition, iB );
    idxBG = find( idxMerge == iB );
    x = getVelocityData( src( idxR ), idxBG ); 
    setData( obj, x, iB, '-v7.3' )
end
