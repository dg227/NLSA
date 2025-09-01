function mkdir( obj ) 
% MKDIR Make directories of nlsaDiffusionOperator_batch objects
%
% Modified 2014/02/07

mkdir@nlsaDiffusionOperator( obj )
for iObj = 1 : numel( obj )
    pth = getNormalizationPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
    pth = getDegreePath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
