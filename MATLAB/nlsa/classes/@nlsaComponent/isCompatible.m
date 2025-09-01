function [ ifC, Test1, Test2 ] = isCompatible( obj1, obj2, varargin )
% ISCOMPATIBLE Check compatibility of nlsaComponent objects 
%
% Modified 2020/03/19

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
        % defaults
        Test2.passNC12 = true;
        Test2.passND12 = NaN;

        if size( obj1, 1 ) ~= size( obj2, 1 )
            Test2.passNC12 = false;
            ifC            = false;
        end

        if Test2.passNC12
            if any( Test1.nD ~= Test2.nD )
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
            if any( getNSample( obj1( 1, : ) ) ~= getNSample( obj2( 1, : ) ) )
                Test2.passNS12 = false;
                ifC = false;
            else
                Test2.passNS12 = true;
            end
        end
    end

elseif isa( obj2, 'nlsaKernelOperator' )

    [ ifC2, Test2 ] = isCompatible( getPartition( obj1( 1, : ) ), ...
                                    getPartitionTest( obj2 ) );
    ifC = ifC && ifC2;

end
