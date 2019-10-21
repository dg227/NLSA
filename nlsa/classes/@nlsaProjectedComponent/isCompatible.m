function ifC = isCompatible( obj1, obj2 )
% ISCOMPATIBLE Check compatibility of nlsaProjectedComponent objects 
%
% Modified 2015/11/18

ifC = true;

% Require that obj1 is a column vector of compatible objects
if ~iscolumn( obj1 )
    ifC = false;
    return
end
for iObj = 2 : numel( obj1 )
    if ~isCompatible( obj1( 1 ), obj1( iObj ) )
        ifC = false;
        return
    end
end

if nargin == 1
    return
end

if isa( obj2, 'projectedComponent' )

        for iObj = 1 : numel( obj1 )
            if ~isCompatible( getPartition( obj1( iObj ) ), ...
                              getPartition( obj2( iObj ) ) )
                ifC = false;
                return
            end
        end


elseif isa( obj2, 'nlsaEmbeddedComponent' )

    % Check compatibility of obj2
    if ~isCompatible( obj2 )
        ifC = false;
        return
    end

    % Require that dimensions are compatible
    nDE1 = getEmbeddingSpaceDimension( obj1 );
    nDE2 = getEmbeddingSpaceDimension( obj2( :, 1 ) );
    if numel( nDE1 ) ~= numel( nDE2 )
        ifC = false;
        return
    end
    if any( nDE1 ~= nDE2 )
        ifC = false;
        return
    end

   % Require that partitions are compatible over all realizations
    for iObj = 1 : numel( obj1 )
        if ~isCompatible( getPartition( obj1( iObj ) ), ...
                          getPartition( obj2( iObj, : ) ) ) 
            ifC = false;
            return
        end
    end

elseif isa( obj2, 'nlsaKernelOperator' )

    % Require that obj1 is a column vector of compatible objects
    if ~iscolumn( obj1 )
        ifC = false;
        return
    end
    for iObj = 2 : numel( obj1 )
        if ~isCompatible( obj1( 1 ), obj1( iObj ) )
            ifC = false;
            return
        end
    end

    % Require that obj2 is scalar
    if ~isscalar( obj2 )
        ifC = false;
        return
    end

    % Require compatibility of partitions
    if ~isCompatible( getPartition( obj1( 1 ) ), ...
        getPartitionTest( obj2 ) )
        ifC = false;
        return
    end

    % Require compatibility of basis function numbers
    if any( getNBasisFunction( obj1 ) > getNEigenfunction( obj2 ) );
        ifC = false;
        return
    end
end

