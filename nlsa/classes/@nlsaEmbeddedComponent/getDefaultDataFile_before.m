function file = getDefaultDataFile_before( obj )
% GETDEFAULTDATAFILE_BEFORE Get default data file before the main time
% interval of an nlsaEmbeddedComponent object
%
% Modified 2014/04/06

nXB = getNXB( obj );
if nXB  > 0
    file = sprintf( 'dataX_%i-0.mat', -( nXB - 1 ) );
else
    file = '';
end

