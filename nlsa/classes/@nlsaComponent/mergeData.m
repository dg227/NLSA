function mergeData( obj, src, idxR )
% MERGEDATA Merge data from a vector of nlsaComponent objects src to 
% a scalar nlsaComponent object obj. obj and src must have the same dimension,
% and the partition of obj must be a coarsening of the merged partitions of 
% src. idxR is an optional input argument specifying the realizations in src
% to be merged. 
%
% Modified 2020/01/25

%% VALIDATE INPUT ARGUMENTS
if ~isa( src, 'nlsaComponent' ) || ~isrow( src )
    error( 'Second argument must be a row vector of nlsaComponent objects.' )
end
nD = getDataSpaceDimension( obj );    
nR = size( src, 2 );
if nargin == 2
    idxR = 1 : nR;
end
if nD ~= getDataSpaceDimension( src( 1 ) )
    error( 'Invalid source data dimension' )
end
partition  = getPartition( obj );
partitionS = getPartition( src( idxR ) );
partitionG = mergePartitions( partitionS );

[ tst, idxMerge ] = isFiner( partitionG, partition );
if ~tst 
    error( 'Incompatible partitions' )
end
nB = getNBatch( partition );

%% LOOP OVER BATCHES OF THE COARSE PARTITION
for iB = 1 : nB
    idxBG = find( idxMerge == iB );
    x = getData( src( idxR ), idxBG ); 
    setData( obj, x, iB, '-v7.3' )
end
