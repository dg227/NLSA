function partition = getEmbPartitionT( obj )
% GETEMBPARTITION Get test partition of the embedded data of an nlsaModel_base
% object
%
% Modified 2019/11/24

component = getEmbComponentT( obj );
partition = getPartition( component( 1, : ) ); 
