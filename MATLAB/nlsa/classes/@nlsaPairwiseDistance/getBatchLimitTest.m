function lim = getBatchLimitTest( obj, iB );
% GETBATCHLIMITTEST  Get batch limits of test data for nlsaPairwiseDistance 
% object 
%
% Modified 2012/04/07

lim = getBatchLimit( getPartitionTest( obj ), iB );
