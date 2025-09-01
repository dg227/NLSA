function [ xiNorm2, xi ] = getVelocity( obj, iBatch, outFormat )
% GETVELOCITY  Read phase space velocity data from an 
% nlsaEmbeddedComponent_xi_o object
%
% Modified 2014/05/26

fileXi = fullfile( getVelocityPath( obj ), ...
                   getVelocityFile( obj, iBatch ) );
load( fileXi, 'xiNorm2' )

if nargout == 1
    return
end
    
if nargin == 2
    outFormat = 'evector';
end

load( fileXi, 'xi' )

if strcmp( outFormat, 'evector' )
    nS      = size( xi, 2 );
    idxE    = getEmbeddingIndices( obj );
    xiNorm2 = lsum( xiNorm2, [ idxE( end ) nS ], idxE );
    xi      = lembed( xi, [ idxE( end ) nS ], idxE );
elseif ~any( strcmp( outFormat, { 'overlap' 'native' } ) )
    error( 'Unrecognized output format' )
end
