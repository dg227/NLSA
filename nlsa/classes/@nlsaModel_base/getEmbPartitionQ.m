function partition = getEmbPartitionQ( obj )
% GETEMBPARTITION Get query partition of the embedded data of an nlsaModel_base
% object
%
% Modified 2019/11/20

partition = getPartition( obj.embComponentQ( 1, : ) ); 
