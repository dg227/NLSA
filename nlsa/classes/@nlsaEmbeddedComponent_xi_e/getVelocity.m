function [ xiNorm2, xi ] = getVelocity( obj, iBatch, outFormat )
% GETVELOCITY  Read phase space velocity data from an nlsaEmbeddedComponent_xi 
% object
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

varList = { 'xiNorm2', 'xi' };
    
fileXi = fullfile( getVelocityPath( obj ), ...
                   getVelocityFile( obj, iBatch ) );
load( fileXi, varList{ 1 : nargout } )
