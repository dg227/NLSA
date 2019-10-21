function obj = setPartition( obj, partition )
% SETPARTITION  Set partition of nlsaDiffusionOperator_batch objects
%
% Modified 2018/06/18

obj = setPartition@nlsaDiffusionOperator( obj, partition );

if ~isCompatible( getOperatorFilelist( obj ), partition )
    fList = nlsaFilelist( partition );
    obj = setOperatorFile( obj, fList );
    obj = setEigenfunctionFile( obj, fList );
    obj = setNormalizationFile( obj, fList );
    obj = setDegreeFile( obj, fList );
end

