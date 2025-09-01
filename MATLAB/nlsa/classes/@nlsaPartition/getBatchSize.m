function nSB = getBatchSize( obj, iB, iR )
% GETBATCHSIZE  Get batch sizes of nlsaPartition object 
%
% Modified 2019/11/10

if nargin == 3
    nSB = getBatchSize( obj( iR ), iB );
    return
end

if numel( obj ) > 1
    obj = mergePartitions( obj );
end

if nargin == 1
    iB = 1 : getNBatch( obj );
end

if isempty( iB )
    nSB = [];
else
    lim = getBatchLimit( obj, iB );
    nSB = lim( :, 2 ) - lim( :, 1 ) + 1;
end
