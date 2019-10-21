function [ ifC, Test1, Test2 ] = isCompatible( obj1, obj2, varargin )
% ISCOMPATIBLE_XI Check compatibility of nlsaEmbeddedComponent_xi objects 
%
% Modified 2015/01/05

[ ifC, Test1 ] = isCompatible_xi( obj1 );

if nargin == 1
    [ ifCParent, Test1Parent ] = isCompatible@nlsaEmbeddedComponent( obj1 );
    ifC   = ifC && ifCParent;
    Test1 = catstruct( Test1, Test1Parent );
    return
end

if isa( obj2, 'nlsaComponent' )

    Opt.testComponents = true;
    Opt.testSamples    = true;
    Opt = parseargs( Opt, varargin{ : } );

    ifXi = isa( obj2, 'nlsaEmbeddedComponent_xi' );
    if ifXi
        [ ifC2, Test2 ] = isCompatible_xi( obj2 );
        ifC = ifC && ifC2;
    end
    [ ifCParent, Test1Parent, Test2Parent ] = isCompatible@nlsaEmbeddedComponent( obj1, obj2, ...
                                              'testComponents', Opt.testComponents, ...
                                              'testSamples', Opt.testSamples );
    ifC   = ifC && ifCParent;
    Test1 = catstruct( Test1, Test1Parent );
    if ifXi
        Test2 = catstruct( Test2, Test2Parent );
    else 
        Test2 = Test2Parent;
    end
    if ifXi && Opt.testComponents
        Test2.passFDOrd12 = true;
        Test2.passFDType12 = true;
        if Test2.passNC12
            if any( Test1.fDOrd ~= Test2.fDOrd )
                Test2.passFDOrd12 = false;
                ifC               = false;
            end

            if ~all( strcmp( Test1.fDType, Test2.fDType ) )
                Test2.passFDType12 = false;
                ifC                = false;
            end
        else
            Test2.passFDOrd12 = NaN;
            Test2.passFDType12 = NaN;
        end
    end

elseif isa( obj2, 'nlsaKernelOperator' )

    [ ifCParent, Test1Parent, Test2 ] = isCompatible@nlsaEmbeddedComponent( obj1, obj2 );
    ifC = ifC && ifCParent;
    Test1 = catstruct( Test1, Test1Parent );

end

