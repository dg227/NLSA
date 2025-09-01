function nSB = getBatchSize( obj, iB, iR )
% GETBATCHSIZE  Get batch sizes of nlsaPairwiseDistance object 
%
% Modified 2014/01/04

if nargin == 2
    iR = 1;
end

partition = getPartition( obj, iR );

if nargin == 1
    nSB = getBatchSize( partition );
elseif any( nargin == [ 2 3 ] )
    nSB = getBatchSize( partition, iB );
end
