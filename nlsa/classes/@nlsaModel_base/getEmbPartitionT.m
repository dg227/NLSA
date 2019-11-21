function partition = getEmbPartitionT( obj )
% GETEMBPARTITION Get test partition of the embedded data of an nlsaModel_base
% object
%
% Modified 2019/11/20

partition = getPartition( obj.embComponentT( 1, : ) ); 
