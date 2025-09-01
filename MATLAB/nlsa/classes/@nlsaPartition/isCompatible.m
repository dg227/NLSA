function [ ifC, Test ] = isCompatible( obj1, obj2 )
% ISCOMPATIBLE Check for compatibility of nlsaPartition objects
%
% Modified 2015/01/06

Test.passNR  = true;
Test.passIdx = true;

ifC = isequal( obj1, obj2 );

if isnan( ifC )
    ifC = false;
    Test.passNR  = false;
    Test.passIdx = NaN; 
end

if Test.passNR
    Test.passIdx = ifC;
end
