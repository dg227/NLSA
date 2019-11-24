function partition = getDenEmbPartitionQ( obj )
% GETDENEMBPARTITIONQ Get query partition of the embedded density data of an 
% nlsaModel_den object
%
% Modified 2019/11/24

component = getDenEmbComponentQ( obj );
partition = getPartition( component( 1, : ) ); 
