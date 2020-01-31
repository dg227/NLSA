function [ res, idxB, idxR ] = isFiner( partition1, partition2 )
%% ISFINER Check for refinement relationship of two scalar nlsaPartition
%% objects
%
% res is a logical variable set to true if partition1 is a refinement of
% partition 2; otherwise it is set to false.
%
% partition1 can be vector or scalar; partition2 must be scalar.  
% 
% if res == true, idxB and idxR are vectors of integers of size equal to the
% total number of batches in partition1, storing their affiliation to the 
% batches of partition2. Otherwise, they are set to empty vectors.
%
% Modified 2019/11/11

% Validate input arguments
if ~isvector( partition1 )
    error( 'First argument must be a scalar or vector nlsaPartition object.' )
end
if ~isa( partition2, 'nlsaPartition' ) || ~isscalar( partition2 )
    error( 'Second arguement must be a scalar nlsaPartition object.' )
end

partitionG = mergePartitions( partition1 );

% Sample numbers
nS1 = getNSample( partitionG );
nS2 = getNSample( partition2 );

% Batch numbers
nB1 = getNBatch( partitionG );
nB2 = getNBatch( partition2 );

% Quick return if partitions have different sample numbers, or partition2
% has more batches than partitionG
if nS1 ~= nS2 || nB2 > nB1
    res = false;
    idxB = [];
    idxR = [];
    return
end

idx1 = getIdx( partitionG );
idx2 = getIdx( partition2 );

if numel( setdiff( idx1, idx2 ) ) ~= nB1 - nB2
    res = false;
    idxB = [];
    idxR = [];
    return
end

res = true;

if nargout >= 2
    idxB = findBatch( partition2, idx1 );
end

if nargout == 3
    [ idxB, idxR ] = gl2loc( partition1, idxG );
end







