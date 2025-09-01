function ifC = isCompatible( obj1, obj2 )
% ISCOMPATIBLE Check compatibility of nlsaCovarianceOperator objects 
%
% Modified 2014/07/16

ifC = true;
switch class( obj2 )
    case 'embeddedComponent'
   
        % Require that the array of embeddd component objects is 
        % self-compatible
        if ~isCompatible( obj2 )
            ifC = false;
            return
        end

        partition  = getPartition( obj1 );
        nR = numel( partition );
        nD = getDimension( obj1 );
        nC = numel( nD );
        siz = size( obj2 );

        % Require that number of components is consistent
        if siz( 1 ) ~= nC
            ifC = false;
            return
        end

        % Require that number of realizations is consistent
        if siz( 2 ) ~= nR
            ifC = false;
            return
        end

        % Require that partitions are consistent
        if any( ~isequal( partition, getPartition( obj2( 1, : ) ) ) )
            ifC = false;
            return
        end

        % Require that dimensions are consistent
        if any( nD ~= getEmbeddingSpaceDimension( obj2( :, 1 ) ) )
            ifC = false;
            return
        end

end

