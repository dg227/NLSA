function nS = getNSampleTest( obj )
% GETNSAMPLETEST  Get the number of test samples in each realization for an 
% nlsaKernelOperator object
%
% Modified 2014/07/16

nS = getNSample( getPartitionTest( obj ) );
