function [ ifC, Test1, Test2 ] = isCompatible( obj1, obj2, obj3 ) 
% ISCOMPATIBLE Check compatibility of nlsaComponent_rec objects 
%
% Modified 2015/08/27

[ ifC, Test1 ] = isCompatible_rec_phi( obj1 );

if nargin == 1
    [ ifCParent, Test1Parent ] = isCompatible@nlsaComponent_phi( obj1 );
    ifC                        = ifC && ifCParent;
    Test1                      = catstruct( Test1, Test1Parent );
    return
end

if isa( obj2, 'nlsaProjectedComponent' ) || isa( obj2, 'nlsaLinearMap' )

    nC = getNComponent( obj1 );
    Test2.passNC12 = nC ~= getNComponent( obj2 )
    if Test.passNC12
        Test.passND12 = true;
        for iC = 1 : nC
            if ~isposint( getEmbeddingSpaceDimension( obj2, iC ) ...
                           / getDimension( obj1( iC, 1 ) ) )
                Test.passND12 = false;
                ifC = false;
                break
            end
        end
    else
        ifC = false;
        Tes.passND12 = NaN;
    end
end

if nargin == 3
    
    ifC = ifC && 


    Opt.testComponents = true;
    Opt.testSamples    = true;
    Opt = parseargs( Opt, varargin{ : } );

    ifE = isa( obj2, 'nlsaEmbeddedComponent' );
    if ifE
        [ ifC2, Test2 ] = isCompatible_emb( obj2 );
        ifC = ifC && ifC2;
    end
    [ ifCParent, Test1Parent, Test2Parent ] = isCompatible@nlsaComponent( obj1, obj2, ...
                                              'testComponents', Opt.testComponents, ... 
                                              'testSamples', Opt.testSamples );
    ifC = ifC && ifCParent;    
    Test1 = catstruct( Test1, Test1Parent );
    if ifE 
        Test2 = catstruct( Test2, Test2Parent );
    else
        Test2 = Test2Parent;
    end
    if ifE && Opt.testComponents
        Test2.passIdxE12 = true;
        if Test2.passNC12
            if ~iscell( Test1.idxE )
                idxE1 = { Test1.idxE };
                idxE2 = { Test2.idxE };
            else
                idxE1 = Test1.idxE;
                idxE2 = Test2.idxE;
            end
            for iC = 1 : numel( idxE1 )
                if any( idxE1{ iC } ~= idxE2{ iC } )
                    Test2.passIdxE12 = false;
                    ifC              = false;
                end
            end
        else
            Test2.passIdxE12 = NaN;
        end
    end

elseif isa( obj2, 'nlsaKernelOperator' )

    [ ifCParent, Test1Parent, Test2 ] = isCompatible@nlsaComponent( obj1, obj2 );
    ifC = ifC && ifCParent;
    Test1 = catstruct( Test1, Test1Parent );

end
