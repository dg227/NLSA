function pth = getVelocityPath( obj )
% GETVELOCITYPATH Get phase space velocity path of an array of 
% nlsaEmbeddedComponent_xi objects
%
% Modified 2014/03/31

pth = getPath( obj );
if isscalar( obj )
    pth = fullfile( pth, getVelocitySubpath( obj ) );
else
    for iObj = 1 : numel( obj )
        pth{ iObj } = fullfile( pth{ iObj }, getVelocitySubpath( obj( iObj ) ) );
    end
end
