function file = getDefaultEigenfunctionFile( obj )
% GETDEFAULTEIGENFUNCTIONFILE Get default eigenfunction file of an nlsaDiffusionOperator_batch object
%
% Modified 2014/04/07

file = getDefaultFile( getEigenfunctionFilelist( obj ), ...
                       getPartition( obj ), 'dataPhi' );
