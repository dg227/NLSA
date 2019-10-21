function xiNorm = getVelocityNorm( obj, iB, outFormat )
% GETVELOCITYNORM  Read phase space velocity norm from an 
% nlsaEmbeddedComponent_xi_e object
%
% Modified 2014/04/05

if nargin == 2
    outFormat = 'evector';
end

if strcmp( outFormat, 'overlap' )
    error( 'Overlap output format not available.' )
end

xiNorm = sqrt( getVelocity( obj, iB, outFormat ) );
