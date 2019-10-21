function mkdir( obj ) 
% MKDIR Make directories of nlsaDiffusionOperator objects
%
% Modified 2014/02/07

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
