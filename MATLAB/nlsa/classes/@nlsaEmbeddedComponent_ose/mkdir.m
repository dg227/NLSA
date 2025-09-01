function mkdir( obj ) 
% MKDIR Make directories of nlsaEmbeddedComponent_ose objects
%
% Modified 2014/05/26

mkdir@nlsaEmbeddedComponent_xi( obj )

for iObj = 1 : numel( obj )
    pth = getVelocityErrorPath( obj( iObj ) ); 
    if ~isdir( pth ) 
        mkdir( pth )
    end
    pth = getStateErrorPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
