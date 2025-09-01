function mkdir( obj ) 
% MKDIR Make directories of nlsaComponent objects
%
% Modified 2015/08/31


for iObj = 1 : numel( obj )
    pth = getDataPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
