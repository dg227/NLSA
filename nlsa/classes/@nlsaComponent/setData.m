function setData( obj, x, iB, varargin )
% SETDATA  Set data of an nlsaComponent object
% varargin is used to pass flags for Matlab's save function 
%
% Modified 2015/10/14

nSB = getBatchSize( obj, iB );
nD = getDimension( obj );

if any( size( x ) ~= [ nD nSB ] )
    error( 'Incompatible size of data array' )
end
    
fileX = fullfile( getDataPath( obj ), ... 
                  getDataFile( obj, iB ) );


save( fileX, 'x', varargin{ : } )


