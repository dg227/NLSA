function partition = getEmbPartitionQ( obj )
% GETEMBPARTITION Get query partition of the embedded data of an nlsaModel_base
% object
%
% Modified 2019/11/24

component = getEmbComponentQ( obj );

partition = getPartition( component( 1, : ) ); 
