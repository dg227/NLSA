function lim = getBatchLimit( obj, iB, iR );
% GETBATCHLIMIT  Get batch limits of nlsaPartition object 
%
% Modified 2017/07/20

if nargin == 3
    lim = getBatchLimit( obj( iR ), iB );
end

if numel( obj ) > 1
    obj = mergePartitions( obj );
end

if nargin == 1
    iB = 1 : getNBatch( obj );
end

nOut = numel( iB );

lim = zeros( nOut, 2 );

if1 = iB == 1;

lim( :, 2 )    = obj.idx( iB )';
lim( ~if1, 1 ) = obj.idx( iB( ~if1 ) - 1 ) + 1;
lim( if1, 1 )  = 1;
