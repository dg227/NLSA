function mkdir( obj ) 
% MKDIR Make directories of nlsaProjection_xi objects
%
% Modified 2014/06/24

mkdir@nlsaProjectedComponent( obj )

for iObj = 1 : numel( obj )
    pth = getVelocityProjectionPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
