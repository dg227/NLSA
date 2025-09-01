function pth = getStateErrorPath( obj )
% GETSTATEERRORPATH Get state error path of an array of 
% nlsaEmbeddedComponent_ose objects
%
% Modified 2014/05/20

pth = getPath( obj );
if isscalar( obj )
    pth = fullfile( pth, getStateErrorSubpath( obj ) );
else
    for iObj = 1 : numel( obj )
        pth{ iObj } = fullfile( pth{ iObj }, ...
                                getStateErrorSubpath( obj( iObj ) ) );
    end
end
