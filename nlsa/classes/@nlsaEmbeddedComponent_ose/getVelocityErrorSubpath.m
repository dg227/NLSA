function pth = getVelocityErrorSubpath( obj )
% GETVELOCITYERRORSUBPATH Get phase space velocity error subpath of an array 
% of nlsaComponent_ose objects
%
% Modified 2014/03/31

pth = cell( size( obj ) );
for iObj = 1 : numel( obj );
    pth{ iObj } = obj( iObj ).pathEXi;
end
            
if isscalar( obj )
    pth = pth{ 1 };
end
