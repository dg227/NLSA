function x = getData( obj, iB, outFormat )
% GETDATA  Read data from an nlsaEmbeddedComponent_o object 
%
% Modified 2014/05/14

if nargin == 2
    outFormat = 'evector';
end
    
if ~any( strcmp( outFormat, { 'overlap' 'evector' 'native' } ) )
    error( 'Unrecognized output format' )
end

fileX = fullfile( getDataPath( obj ), getDataFile( obj, iB ) );
load( fileX, 'x' )


if strcmp( outFormat, 'evector' ) 
    idxE = getEmbeddingIndices( obj );
    x = lembed( x, [ idxE( end ) size( x, 2 ) ], idxE );
end
