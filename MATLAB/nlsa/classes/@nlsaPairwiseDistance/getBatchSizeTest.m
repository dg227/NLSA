function nSB = getBatchSizeTest( obj, iB, iR )
% GETBATCHSIZETEST  Get batch sizes of nlsaPairwiseDistance object 
%
% Modified 2014/04/03

if nargin == 2
    iR = 1;
end

partition = getPartitionTest( obj, iR );

if nargin == 1
    nSB = getBatchSize( partition );
elseif any( nargin == [ 2 3 ] )
    nSB = getBatchSize( partition, iB );
end
