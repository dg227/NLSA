function file = getDefaultOperatorFile( obj )
% GETDEFAULTOPERATORFILE Get default operator file of an nlsaDiffusionOperator_batch object
%
% Modified 2014/04/07

file = getDefaultFile( getOperatorFilelist( obj ), ...
                       getPartition( obj ), 'dataP' );
