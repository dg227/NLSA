function mergeData( obj, src )
% MERGEDATA Merge data from a vector of nlsaComponent objects src to 
% a scalar nlsaComponent object obj. obj and src must have the same dimension,
% and the partition of obj must be a coarsening of the merged partitions of 
% src.
%
% Modified 2019/11/10

%% VALIDATE INPUT ARGUMENTS
nD = getDimension( obj );    
partition  = getPartition( obj );
partitionS = getPartition( src ) );
partitionG = mergePartitions( partitionS );
if nD ~= getDimension( src( 1 ) )
    error( 'Invalid source data dimension' )
end
[ tst, idxMerge ] = iscoarser( partition, partitionG );
if ~tst 
    error( 'IncompatiblePartitions' )
end
nB = getNBatch( partition );

%% LOOP OVER BATCHES OF THE COARSE PARTITION


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
% Read the source batch containing the start index, taking into 
% account extra samples needed for embedding and/or nXB.
% Below, iSBSrc1 is the batch-local index in the source data
iWant   = obj.idxO - ( nE - 1 + obj.nXB );
iBSrc   = findBatch( src, iWant, iR );
xSrc    = getData( src, iBSrc, iR );
lSrc    = getBatchLimit( src, iBSrc, iR );
nSBSrc  = getBatchSize( src, iBSrc, iR );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAMPLES BEFORE MAIN INTERVAL
if obj.nXB > 0
    iSBE1 = 1; % iSBE is the batch-local index in the embedded data
    nSBE  = obj.nXB;
    x     = zeros( nDE, nSBE );
    iSBSrc1 = obj.idxO - lSrc( 1 ) - nSBE + 1;
    deficit = nSBE;
    while deficit >= 0 
        nSProvided = min( nSBSrc - iSBSrc1 + 1, nSBE - iSBE1 + 1 );
        iSBSrc2    = iSBSrc1 + nSProvided - 1;
        iSBE2      = iSBE1 + nSProvided - 1;
        x( :, iSBE1 : iSBE2 ) = lembed( xSrc, [ iSBSrc1 iSBSrc2 ], obj.idxE );
        iSBE1   = iSBE2 + 1;
        iSBSrc1 = iSBSrc2 + 1;        
        deficit = nSBE - iSBE1;
        if deficit >= 0 && iSBSrc1 > nSBSrc
            iBSrc   = iBSrc + 1;
            iKeep1  = nSBSrc - obj.idxE( end ) + 2;
            iKeep2  = nSBSrc; 
            xKeep   = xSrc( :, iKeep1 : iKeep2 );
            xSrc    = [ xKeep getData( src, iBSrc, iR ) ];
            nSKeep  = size( xKeep, 2 );
            iSBSrc1 = 1 + nSKeep; 
            nSBSrc  = getBatchSize( src, iBSrc, iR ) + nSKeep;
        end
    end
    setData_before( obj, x, '-v7.3' )
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN-INTERVAL SAMPLES
% Loop over the embedded data batches
iSBSrc1 = obj.idxO - lSrc( 1 ) + 1;      
for iBE = 1 : getNBatch( obj )
    iSBE1 = 1; % iSBE is the batch-local index in the embedded data
    nSBE = getBatchSize( obj, iBE );
    x     = zeros( nDE, nSBE );
    deficit = nSBE;
    while deficit >= 0 
        nSProvided = min( nSBSrc - iSBSrc1 + 1, nSBE - iSBE1 + 1 );
        iSBSrc2    = iSBSrc1 + nSProvided - 1;
        iSBE2      = iSBE1 + nSProvided - 1;
        x( :, iSBE1 : iSBE2 ) = lembed( xSrc, [ iSBSrc1 iSBSrc2 ], obj.idxE );
        iSBE1   = iSBE2 + 1;
        iSBSrc1 = iSBSrc2 + 1;        
        deficit = nSBE - iSBE1;
        if deficit >= 0 && iSBSrc1 > nSBSrc
            iBSrc   = iBSrc + 1;
            iKeep1  = nSBSrc - obj.idxE( end ) + 2;
            iKeep2  = nSBSrc; 
            xKeep   = xSrc( :, iKeep1 : iKeep2 );
            xSrc    = [ xKeep getData( src, iBSrc, iR ) ];
            nSKeep  = size( xKeep, 2 );
            iSBSrc1 = 1 + nSKeep; 
            nSBSrc  = getBatchSize( src, iBSrc, iR ) + nSKeep;
        end
    end
    setData( obj, x, iBE, '-v7.3' )
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAMPLES AFTER MAIN INTERVAL
if obj.nXA > 0
    iSBE1 = 1; % iSBE is the batch-local index in the embedded data
    nSBE  = obj.nXA;
    x     = zeros( nDE, nSBE );
    deficit = nSBE;
    while deficit >= 0 
        nSProvided = min( nSBSrc - iSBSrc1 + 1, nSBE - iSBE1 + 1 );
        iSBSrc2    = iSBSrc1 + nSProvided - 1;
        iSBE2      = iSBE1 + nSProvided - 1;
        x( :, iSBE1 : iSBE2 ) = lembed( xSrc, [ iSBSrc1 iSBSrc2 ], obj.idxE );
        iSBE1   = iSBE2 + 1;
        iSBSrc1 = iSBSrc2 + 1;        
        deficit = nSBE - iSBE1;
        if deficit >= 0 && iSBSrc1 > nSBSrc
            iBSrc   = iBSrc + 1;
            iKeep1  = nSBSrc - obj.idxE( end ) + 2;
            iKeep2  = nSBSrc; 
            xKeep   = xSrc( :, iKeep1 : iKeep2 );
            xSrc    = [ xKeep getData( src, iBSrc, iR ) ];
            nSKeep  = size( xKeep, 2 );
            iSBSrc1 = 1 + nSKeep; 
            nSBSrc  = getBatchSize( src, iBSrc, iR ) + nSKeep;
        end
    end
    setData_after( obj, x, '-v7.3' )
end  
