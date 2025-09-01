function mkdir( obj ) 
% MKDIR Make directories of nlsaSymmetricDistance objects
%
% Modified 2014/05/01

for iObj = 1 : numel( obj )
    pth = getDistancePath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
