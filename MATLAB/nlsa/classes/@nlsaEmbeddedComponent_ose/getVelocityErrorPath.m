function pth = getVelocityErrorPath( obj )
% GETVELOCITYERRORPATH Get phase space velocity error path of an array of 
% nlsaEmbeddedComponent_ose objects
%
% Modified 2014/03/31

pth = getPath( obj );
if isscalar( obj )
    pth = fullfile( pth, getVelocityErrorSubpath( obj ) );
else
    for iObj = 1 : numel( obj )
        pth{ iObj } = fullfile( pth{ iObj }, ...
                                getVelocityErrorSubpath( obj( iObj ) ) );
    end
end
