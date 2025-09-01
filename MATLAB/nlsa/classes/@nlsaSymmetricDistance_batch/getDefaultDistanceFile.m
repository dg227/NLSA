function file = getDefaultDistanceFile( obj )
% GETDEFAULTFILE Get default files of an nlsaSymmetricDistance_batch object
%
% Modified 2014/04/30

file = getDefaultFile( getDistanceFilelist( obj ), ...
                       getPartition( obj ), 'dataYS' );
