function mkdir( obj ) 
% MKDIR Make directories of nlsaCovarianceOperator objects
%
% Modified 2014/07/16

for iObj = 1 : numel( obj )
    pth = getOperatorPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
    pth = getLeftSingularVectorPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
    pth = getRightSingularVectorPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
    pth = getSingularValuePath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
