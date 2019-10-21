function partition = getPartition( obj, iR )
% GETPARTITION  Get partition of an nlsaProjected component object
%
% Modified 2014/06/24

switch nargin
    case 1
        partition = obj.partition;
    case 2
        partition = obj.partition( iR );
end
