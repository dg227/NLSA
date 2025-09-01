function xiNorm = getVelocityNorm( obj, iB, outFormat )
% GETVELOCITYNORM  Read phase space velocity norm from an 
% nlsaEmbeddedComponent_xi_o object
%
% Modified 2014/05/14

xiNorm = getVelocity( obj, iB );

if nargin == 2
    outFormat = 'evector';
end

if strcmp( outFormat, 'evector' )
    xiNorm = lsum( xiNorm, [ getEmbeddingWindow( obj ) numel( xiNorm ) ], ...
                           getEmbeddingIndices( obj ) );
end

xiNorm = sqrt( xiNorm );

