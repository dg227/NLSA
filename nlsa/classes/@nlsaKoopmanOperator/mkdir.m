function mkdir( obj ) 
% MKDIR Make directories of nlsaKoopmanOperator objects
%
% Modified 2020/04/15

for iObj = 1 : numel( obj )
    pth = getOperatorPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
    pth = getEigenfunctionPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
