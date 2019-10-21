function pth = getStateErrorSubpath( obj )
% GETSTATEERRORSUBPATH Get state error subpath of an array of nlsaComponent_ose 
% objects
%
% Modified 2014/05/26

pth = cell( size( obj ) );
for iObj = 1 : numel( obj );
    pth{ iObj } = obj( iObj ).pathEX;
end
            
if isscalar( obj )
    pth = pth{ 1 };
end
