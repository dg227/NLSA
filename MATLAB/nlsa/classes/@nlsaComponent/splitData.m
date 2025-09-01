function splitData( obj, src )
% SPLITDATA Split data from a scalar nlsaRoot object src to a row vector
% of nlsaComponent objects obj. obj and src must have the same dimension,
% and the partition of src must be a coarsening of the merged partitions of 
% obj. 
%
% Modified 2020/04/04

%% VALIDATE INPUT ARGUMENTS
if ~isrow( obj )
    error( 'First argument must be a row vector of nlsaComponent objects.' )
end
if ~isscalar( src )
    error( 'Second argument must be a scalar.' )
end
nD = getDataSpaceDimension( obj( 1 ) );    
nR = size( obj, 1 );
if nD ~= getDataSpaceDimension( src )
    error( 'Invalid source data dimension' )
end

%% VALIDATE INPUT PARTITIONS AND EXTRACT BATCH INDICES
% idxMerge is a row vector of size equal to the total batches in the output obj. 
% idxMerge( i ) indicates the batch index of the source partition from witch batch i of the 
% global output partition receives data.  
partition            = getPartition( obj );
[ partitionG, idxG ] = mergePartitions( partition );
partitionS           = getPartition( src );
[ tst, idxMerge ] = isFiner( partitionG, partitionS );
if ~tst 
    error( 'Incompatible partitions' )
end

%% LOOP OVER BATCHES OF THE COARSE PARTITION
nBS = getNBatch( partitionS );
for iBS = 1 : nBS
    % Batch iBS of the coarse partition will be split into batches idxBG of the fine partition
    x = getData( src, iBS );
    idxBG = find( idxMerge == iBS );  
    nBG = numel( idxBG );
    iS1 = 1;
    for iBG = 1 : nBG 
        iS2 = iS1 + getBatchSize( partitionG, idxBG( iBG ) ) - 1;
        setData( obj( idxG( 1, idxBG( iBG ) ) ), x( :, iS1 : iS2 ), ...
            idxG( 2, idxBG( iBG ) ), '-v7.3' )
        iS1 = iS2 + 1;
    end 
end
