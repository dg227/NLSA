function obj = setEpsilonTest( obj, epsilon )
% SETEPSILONTEST  Set bandwidth parameter for the test data 
% of an nlsaDiffusionOperator object
%
% Modified 2016/01/25

if ~isps( epsilon )
    error( 'The bandwidth parameter for the test data must be a positive scalar integer' )
end
obj.epsilonT = epsilonT;
