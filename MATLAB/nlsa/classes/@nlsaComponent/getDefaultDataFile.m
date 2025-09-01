function file = getDefaultDataFile( obj )
% GETDEFAULTDATAFILE Get default data files of nlsaComponent object
%
% Modified 2014/04/06

file = getDefaultFile( getDataFilelist( obj ), getPartition( obj ), 'dataX' );
