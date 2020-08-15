function [ ifC, Test, Test2 ] = isCompatible_emb( obj1, obj2, varargin )
% ISCOMPATIBLE_EMB Check compatibility of nlsaEmbeddedComponent objects 
%
% Modified 2020/08/02

ifC            = true;

[ nC1, nR1 ] = size( obj1 );

Test.idxO     = getOrigin( obj1( 1, : ) );
Test.idxE     = getEmbeddingIndices( obj1( :, 1 ) );
Test.passIdxO = true;
Test.passIdxE = true;


% Check that embedding origins are compatible for each component  
for iC = 2 : nC1
    if any(  getOrigin( obj1( iC, : ) ) ~= Test.idxO );
        Test.passIdxO = false;
        ifC            = false;
        break
    end
end


% Check that embedding indices are compatible for each realization	
for iR = 2 : nR1
    if nC1 == 1
        if any( Test.idxE ~= getEmbeddingIndices( obj1( :, iR ) ) )
            Test.passIdxE  = false;
            ifC             = false;
            break
        end
    else 
        for iC = 1 :nC1
            if any( Test.idxE{ iC } ~= getEmbeddingIndices( obj1( iC, iR ) ) )
                Test.passIdxE  = false;
                ifC             = false;
                break
            end
        end
    end 
end

if nargin == 1
    return
end

Opt.testComponents = true;
Opt.testSamples    = true;
Opt = parseargs( Opt, varargin{ : } );

[ ifC2, Test2 ] = isCompatible_comp( obj2 );
ifC             = ifC && ifC2;

if Opt.testComponents
    % defaults
    Test2.passNC12 = true;
    Test2.passND12 = NaN;

    if size( obj1, 1 ) ~= size( obj2, 1 )
        Test2.passNC12 = false;
        ifC            = false;
    end

    if Test2.passNC12
        if any( getDimension( obj1( :, 1 ) ) ~= Test2.nD )
            Test2.passND12 = false;
            ifC = false;
        else
            Test2.passND12 = true;
        end
    end
end

if Opt.testSamples
    % defaults
    Test2.passNR12 = true;
    Test2.passNS12 = NaN;

    if size( obj1, 2 ) ~= size( obj2, 2 )
        Test2.passNR12 = false;
        ifC           = false;
    end

    if Test2.passNR12
        nSE = getNSample( obj1( 1, : ) );
        nXA = getNXA( obj1( 1, : ) );
        nE = max( Test.idxO ) - 1; 
        nSTot = nSE + nXA + nE;
        if any( nSTot > getNSample( obj2( 1, : ) ) )
            Test2.passNS12 = false;
            ifC = false;
        else
            Test2.passNS12 = true;
        end
    end
end
