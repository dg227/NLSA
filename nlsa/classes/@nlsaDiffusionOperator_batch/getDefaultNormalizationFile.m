function file = getDefaultNormalizationFile( obj )
% GETDEFAULTNORMALIZATIONFILE Get default normalization file of an 
% nlsaDiffusionOperator_batch object
%
% Modified 2018/06/18

file = getDefaultFile( getNormalizationFilelist( obj ), ...
                       getPartition( obj ), 'dataQ' );
