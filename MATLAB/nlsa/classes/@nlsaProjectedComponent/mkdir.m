function mkdir( obj ) 
% MKDIR Make directories of nlsaProjection objects
%
% Modified 2014/06/20

for iObj = 1 : numel( obj )
    pth = getProjectionPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
