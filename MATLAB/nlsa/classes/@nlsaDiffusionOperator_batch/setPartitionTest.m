function obj = setPartitionTest( obj, partition )
% SETPARTITIONTEST  Set test partition of nlsaDiffusionOperator_batch objects
%
% Modified 2018/06/18

obj = setPartitionTest@nlsaDiffusionOperator( obj, partition );

if ~isCompatible( getNormalizationFilelist( obj ), partition ) 
    filelist = nlsaFilelist( partition );
%    obj = setNormalizationFile( obj, filelist );
end

