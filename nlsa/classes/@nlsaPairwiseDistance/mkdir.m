function mkdir( obj ) 
% MKDIR Make directories of nlsaPairwiseDistance objects
%
% Modified 2014/10/13

for iObj = 1 : numel( obj )
    pth = getDistancePath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth );
    end
end
