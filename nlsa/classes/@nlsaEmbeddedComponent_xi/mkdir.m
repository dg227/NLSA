function mkdir( obj ) 
% MKDIR Make directories of nlsaEmbeddedComponent_xi objects
%
% Modified 2013/03/31

mkdir@nlsaEmbeddedComponent( obj )

for iObj = 1 : numel( obj )
    pth = getVelocityPath( obj( iObj ) ); 
    if ~isdir( pth ) 
        mkdir( pth )
    end
end
