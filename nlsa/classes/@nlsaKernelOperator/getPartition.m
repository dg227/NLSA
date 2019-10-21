function partition = getPartition( obj, iR )
% GETPARTITION  Get partition of an nlsaDiffusionOperator object
%
% Modified 2014/07/16

switch nargin
    case 1
        partition = obj.partition;
    case 2
        partition = obj.partition( iR );
end
