function partition = getOutEmbPartition( obj )
% GETOSEEMBPARTITION Get partition of the embedded OS data of an 
% nlsaModel_den_ose object
%
% Modified 2018/07/01

comp = getOutEmbComponent( obj );
partition = getPartition( comp( 1, : ) ); 
