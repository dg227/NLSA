function [ partition, idxG ] = mergePartitions( obj );
% MERGEPARTITIONS Merge the partitions form an array of nlsaComponent objects into a single nlsaPartition 
%
% Modified 2013/12/19

partition = mergePartitions( getPartition( obj ) );
