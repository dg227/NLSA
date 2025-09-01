function setData( obj, x, iB, varargin )
% SETDATA  Set data of an nlsaComponent object
% varargin is used to pass flags for Matlab's save function 
%
% Modified 2020/01/25

[ siz( 1 ), siz( 2 ) ] = getBatchArraySize( obj, iB ); 
sizX = size( x );
if ~ismatrix( x ) || any( sizX ~= siz )
    error( [ 'Incompatible size of data array. ' ...
             'Expecting ' int2str( siz ) '. ' ...
             'Received ' int2str( sizX ) '.' ] )
end
    
fileX = fullfile( getDataPath( obj ), ... 
                  getDataFile( obj, iB ) );

save( fileX, 'x', varargin{ : } )


