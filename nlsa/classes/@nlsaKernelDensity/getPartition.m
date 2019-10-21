function partition = getPartition( obj, iR )
% GETPARTITION  Get partition of an nlsaKernelDensity object
%
% Modified 2015/04/06

switch nargin
    case 1
        partition = obj.partition;
    case 2
        partition = obj.partition( iR );
end
