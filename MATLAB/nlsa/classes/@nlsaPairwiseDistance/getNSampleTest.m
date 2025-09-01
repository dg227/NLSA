function nS = getNSampleTest( obj )
% GETNSAMPLETEST  Get number of samples of test data in nlsaPairwiseDistance 
% object
%
% Modified  2012/04/07

nS = getNSample( getPartitionTest( obj ) );
