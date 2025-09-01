function [ ifC, Test1, Test2 ] = isCompatible( obj1, obj2, varargin ) 
% ISCOMPATIBLE Check compatibility of nlsaEmbeddedComponent objects 
%
% Modified 2020/08/02

[ ifC, Test1 ] = isCompatible_emb( obj1 );

if nargin == 1
    [ ifCParent, Test1Parent ] = isCompatible@nlsaComponent( obj1 );
    ifC                        = ifC && ifCParent;
    Test1                      = catstruct( Test1, Test1Parent );
    return
end

if isa( obj2, 'nlsaComponent' )

    Opt.testComponents = true;
    Opt.testSamples    = true;
    Opt = parseargs( Opt, varargin{ : } );

    ifE = isa( obj2, 'nlsaEmbeddedComponent' );
    if ifE
        % If obj2 is an nlsaEmbeddedComponent object, and testSamples is true,
        % we require that the samples of obj1 and obj2 match exactly. 
        [ ifC2, Test2 ] = isCompatible_emb( obj2 );
        ifC = ifC && ifC2;
        [ ifCParent, Test1Parent, Test2Parent ] = ...
            isCompatible@nlsaComponent( obj1, obj2, ...
                                       'testComponents', Opt.testComponents, ...
                                       'testSamples', Opt.testSamples );
    else
        % If obj2 is not an nlsaEmbeddedComponent object, we require that
        % the samples of obj2 are greater than or equal to the samples of 
        % obj1 plus extra samples for delay embedding and finite differencing.
        [ ifCParent, Test1Parent, Test2Parent ] = ...
            isCompatible_emb( obj1, obj2, ...
                              'testComponents', Opt.testComponents, ...
                              'testSamples', Opt.testSamples );
    end

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
