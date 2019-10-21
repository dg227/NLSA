function ifC = isCompatible( obj1, obj2 )
% ISCOMPATIBLE Check compatibility of nlsaLinearMap objects 
%
% Modified 2014/08/10

ifC = true;

if nargin == 1
    % Require that obj1 is a vector of compatible objects
    if ~isvector( obj1 )
        ifC = false;
        return
    end
    for iObj = 2 : numel( obj1 )
        if ~isCompatible( obj1( iObj ), obj1( iObj - 1 ) )
            ifC  = false;
            return
        end
    end
    return
end

switch class( obj2 )

    case 'linearMap'
        % Require that obj2 has a subset of the basis function indices
        % of obj1
        idxPhi1 = getBasisFunctionIndices( obj1 );

        if ~all( idxPhi( 1 : end - 1 ) == getBasisFunctionIndices( obj2 ) )
            ifC = false;
            return
        end

    case 'projectedComponent'

        % Require that obj2 is an array of compatible projected components
        if ~isCompatible( obj2 )
            ifC = false;
            return
        end
        
        % Require that dimensions are compatible
        nDE1 = getDimension( obj1 ); 
        nDE2 = getEmbeddingSpaceDimension( obj2 );
        if numel( nDE1 ) ~= numel( nDE2 )
            ifC = false;
            return
        end
        if any( nDE1 ~= nDE2 )
            ifC = false;
            return
        end

        % Require compatibility of basis function indives
        if any( max( getBasisFunctionIndices( obj1 ) ) ...
          > getNBasisFunction( obj2 ) );
            ifC = false;
            return
        end

    case 'nlsaKernelOperator'
        ifC = true;

        % Require that partitions are compatible
        if ~isCompatible( getPartition( obj1 ), getPartition( obj2 ) )
            ifC = false;
            return
        end

        % Require that the number of basis functions is compatible
        if max( getBasisFunctionIndices( obj1 ) ) ...
          > getNEigenfunction( obj2 ) 
            ifC = false;
            return
        end

end

