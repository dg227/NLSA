function setData( obj, x, iB, varargin )
% SETDATA  Set batch data of an nlsaEmbeddedComponent object 
%
% Modified 2014/04/14

[ siz( 1 ), siz( 2 ) ] = getBatchArraySize( obj, iB ); 
if any( size( x ) ~= siz )
    error( 'Incompatible size of data array' )
end
    
fileX = fullfile( getDataPath( obj ), ... 
                  getDataFile( obj, iB ) );



save( fileX, 'x', varargin{ : } )

