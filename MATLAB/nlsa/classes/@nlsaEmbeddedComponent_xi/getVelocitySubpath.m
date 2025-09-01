function pth = getVelocitySubpath( obj )
% GETVELOCITYSUBPATH Get phase space velocity subpath  of an array 
% of nlsaComponent_xi objects
%
% Modified 2014/03/31

pth = cell( size( obj ) );
for iObj = 1 : numel( obj );
    pth{ iObj } = obj( iObj ).pathXi;
end
            
if isscalar( obj )
    pth = pth{ 1 };
end
