function splitData( obj, src )
% SPLITDATA Split data from a scalar nlsaComponent object src to a row vector
% of nlsaComponent objects obj. obj and src must have the same dimension,
% and the partition of src must be a coarsening of the merged partitions of 
% obj. 
%
% Modified 2020/01/25

%% VALIDATE INPUT ARGUMENTS
if ~isrow( obj )
    error( 'First argument must be a row vector of nlsaComponent objects.' )
end
if ~isa( src, 'nlsaComponent' ) || ~isscalar( src )
    error( 'Second argument must be a scalar nlsaComponent object.' )
end
nD = getDataSpaceDimension( obj( 1 ) );    
nR = size( obj, 1 );
if nD ~= getDataSpaceDimension( src )
    error( 'Invalid source data dimension' )
end
partition            = getPartition( obj );
[ partitionG, idxG ] = mergePartitions( obj );
partitionS           = getPartition( src );
[ tst, idxMerge ] = isFiner( partitionG, partition );
if ~tst 
    error( 'Incompatible partitions' )
end

%% LOOP OVER BATCHES OF THE COARSE PARTITION
nBS = numel( partitionS );
for iBS = 1 : nBS
    xS = getData( src, iBS );
    idxBG = find( idxMerge == iBS );  
    nBG = numel( idxBG );
    iS1 = 1;
    for iBG = 1 : nBG 
        iS2 = iS1 + getBatchSize( partitionG, iBG ) - 1;
        setData( obj( idxG( 1, idxBG( iBG ) ) ), x( :, iS1 : iS2 ), ...
            idxG( 2, idxBG( iBG ) ), '-v7.3' )
    end 
end
