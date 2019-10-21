function partition = getPartition( obj )
% GETPARTITION Get component of an nlsaLocalDistanceData object
%
% Modified 2015/10/26

comp = getComponent( obj );
partition = getPartition( comp( 1, : ) );

