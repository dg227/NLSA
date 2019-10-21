function iB = findBatch( obj, smpl, iR )
% FINDBATCH  Find batch indices of samples in a partition 
%
% Modified 2017/07/20

if nargin == 3
    iB = findBatch( obj( iR ), smpl );
end

if numel( obj ) > 1
    obj = mergePartitions( obj );
end

iB  = zeros( size( smpl ) );
lim = getBatchLimit( obj ) ;
for iS = 1 : numel( smpl )
    iB( iS ) = find( smpl( iS ) >= lim( :, 1 ), 1, 'last' );
end
