function file = getDefaultDataFile_after( obj )
% GETDEFAULTDATAFILE_AFTER Get default data file after the main time
% interval of an nlsaEmbeddedComponent object
%
% Modified 2014/03/30

nXA = getNXA( obj );
if nXA > 0
    nS    = getNSample( obj );
    file = sprintf( 'dataX_%i-%i', nS + 1, nS + nXA );
else
    file = '';
end

