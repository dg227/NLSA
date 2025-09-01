function [ ifC, Test ] = isCompatible_xi( obj )
% ISCOMPATIBLE_XI Check compatibility of nlsaEmbeddedComponent_xi objects 
%
% Modified 2020/08/02

[ ~, nR ]       = size( obj );
Test.fDOrd      = getFDOrder( obj( :, 1 ) );
Test.fDType     = getFDType( obj( :, 1 ) );
Test.passFDord  = true;
Test.passFDType = true;
ifC             = true;


% Check that finite-difference order and type are the same for each realization
for iR = 2 : nR
    if any( Test.fDOrd ~= getFDOrder( obj( :, iR ) ) )
        Test.passFDord = false;
        ifC             = false;
        break
    end
end
for iR = 2 : nR
    if any( ~strcmp( Test.fDType, getFDType( obj( :, iR ) ) ) )
        Test.passFDType = false;
        ifC              = false;
        break
    end
end

