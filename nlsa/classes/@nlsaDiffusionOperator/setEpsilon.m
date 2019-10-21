function obj = setEpsilon( obj, epsilon )
% SETEPSILON  Set bandwidth parameter of an nlsaDiffusionOperator object
%
% Modified 2016/01/25

if ~isps( epsilon )
    error( 'The bandwidth parameter must be a positive scalar integer' )
end
obj.epsilon = epsilon;
