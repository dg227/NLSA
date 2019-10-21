function partition = getPartition( obj, iR )
% GETPARTITION  Get partition of nlsaPairwiseDistance object
%
% Modified 2014/01/06

switch nargin
    case 1
        partition = obj.partition;
    case 2
        partition = obj.partition( iR );
end
