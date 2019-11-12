function [ res, idx ] = isFiner( partition1, partition2 )
%% ISFINER Check for refinement relationship of two scalar nlsaPartition
%% objects
%
% res is a logical variable set to true if partition1 is a refinement of
% partition 2; otherwise it is set to false
%
% if res == true, idx is a vector of integers of size equal to the number of 
% batches in partition 1, storing their affiliation to the batches of 
% partition 2; otherwise, it is set to the empty vector
%
% Modified 2019/11/11

% Validate input arguments
if ~isa( partition2, 'nlsaPartition' )
    error( 'Second arguement must be an nlsaPartition object.' )
end
if ~isscalar( partition1 ) || ~isscalar( partition2 )
    error( 'isFiner accepts only scalar input arguments' )
end

% Sample numbers
nS1 = getNSample( partition1 );
nS2 = getNSample( partition2 );

% Batch numbers
nB1 = getNBatch( partition1 );
nB2 = getNbatch( partition2 );

% Quick return if partitions have different sample numbers, or partition2
% has more batches than partition1
if nS1 ~= nS2 || nB2 > nB1
    res = false;
    idx = [];
    return
end

idx1 = getIdx( partition1 );
idx2 = getIdx( partition2 );

if numel( setdiff( idx1, idx2 ) ) ...
    ~= getNBatch( partition1 ) - getNBatch( partition2 )
    res = false;
    idx = [];
    return
end

res = true;

if nargout == 2
    idx = findBatch( partition2, idx1 );
end





