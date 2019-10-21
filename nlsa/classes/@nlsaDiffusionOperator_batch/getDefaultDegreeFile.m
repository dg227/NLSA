function file = getDefaultDegreeFile( obj )
% GETDEFAULTDEGREEFILE Get default degree file of an 
% nlsaDiffusionOperator_batch object
%
% Modified 2016/01/26

file = getDefaultFile( getDegreeFilelist( obj ), ...
                       getPartition( obj ), 'dataD' );
