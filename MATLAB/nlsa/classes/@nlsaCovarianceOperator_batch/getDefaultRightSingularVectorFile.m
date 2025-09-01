function file = getDefaultRightSingularVectorFile( obj )
% GETDEFAULTRIGHTSINGULARVECTORFILE Get default right singular vector file of 
% an nlsaDiffusionOperator_batch object
%
% Modified 2014/07/16

file = getDefaultFile( getRightSingularVectorFilelist( obj ), ...
                       getPartition( obj ), 'dataV' );
