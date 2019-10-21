function [ ifC, Test1, Test2 ] = isCompatible( obj1, obj2, varargin )
% ISCOMPATIBLE Check compatibility of nlsaComponent objects 
%
% Modified 2015/01/05

[ ifC, Test1 ] = isCompatible_comp( obj1 );

if nargin == 1
    return
end


if isa( obj2, 'nlsaComponent' )

    Opt.testComponents = true;
    Opt.testSamples    = true;
    Opt = parseargs( Opt, varargin{ : } );

    [ ifC2, Test2 ] = isCompatible_comp( obj2 );
    ifC             = ifC && ifC2;

    if Opt.testComponents
        Test2.passNC12 = true;
        Test2.passND12 = true;
    end
    if Opt.testSamples
        Test2.passNR12 = true;
        Test2.passNS12 = true;
    end

    if Opt.testComponents
        if size( obj1, 1 ) ~= size( obj2, 1 )
            Test2.passNC12 = false;
            ifC            = false;
            if Test2.passNC12
                if any( Test1.nD ~= Test2.nD )
                    Test2.passND12 = false;
                    ifC = false;
                end
            else
                Test2.passND12 = NaN;
            end
        end
    end

    if Opt.testSamples
        if size( obj1, 2 ) ~= size( obj2, 2 )
            Test.passNR12 = false;
            ifC           = false;
            if Test.passNR12
                if any( getNSample( obj1( 1, : ) ) ~= getNSample( obj2( 1, : ) ) )
                    Test2.passNS12 = false;
                end
            else
                Test2.passNS12 = NaN;
            end
        end
    end

elseif isa( obj2, 'nlsaKernelOperator' )

    [ ifC2, Test2 ] = isCompatible( getPartition( obj1( 1, : ) ), ...
                                    getPartitionTest( obj2 ) );
    ifC = ifC && ifC2;

end
