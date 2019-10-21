function file = getDataFile( obj, iB )
% GETDATAFILE Get data file of nlsaComponent 
%
% Modified 2014/04/04

if nargin == 1
    iB = 1 : getNBatch( obj );
end

file = getFile( getDataFilelist( obj ), iB ); 
