function x = getData( obj, iB, outFormat )
% GETDATA  Read data from an nlsaEmbeddedComponent_e object
%
% Modified 2014/04/15

if nargin == 2
    outFormat = 'evector'; 
end

if strcmp( outFormat, 'overlap' )
    error( 'Overlap output format not available' )
elseif ~any( strcmp( outFormat, { 'evector' 'native' } ) )
    error( 'Unrecognized output format' )
end

fileX = fullfile( getDataPath( obj ), getDataFile( obj, iB ) );
load( fileX, 'x' )
