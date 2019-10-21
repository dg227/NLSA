function [ ifC, Test ] = isCompatible_emb( obj )
% ISCOMPATIBLE_EMB Check compatibility of nlsaEmbeddedComponent objects 
%
% Modified 2016/12/06

ifC            = true;

[ nC1, nR1 ] = size( obj );

Test.idxO     = getOrigin( obj( 1, : ) );
Test.idxE     = getEmbeddingIndices( obj( :, 1 ) );
Test.passIdxO = true;
Test.passIdxE = true;


% Check that embedding origins are compatible for each component  
for iC = 2 : nC1
    if any(  getOrigin( obj( iC, : ) ) ~= Test.idxO );
        Test.passIdxO = false;
        ifC            = false;
        break
    end
end


% Check that embedding indices are compatible for each realization	
for iR = 2 : nR1
    if nC1 == 1
        if any( Test.idxE ~= getEmbeddingIndices( obj( :, iR ) ) )
            Test.passIdxE  = false;
            ifC             = false;
            break
        end
    else 
        for iC = 1 :nC1
            if any( Test.idxE{ iC } ~= getEmbeddingIndices( obj( iC, iR ) ) )
                Test.passIdxE  = false;
                ifC             = false;
                break
            end
        end
    end 
end
