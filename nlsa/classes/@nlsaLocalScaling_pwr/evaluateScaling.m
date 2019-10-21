function s = evaluateScaling( obj, I )
%% EVALUATESCALING  Evaluate density power law scaling 
%
% Modified 2015/01/05

p = getExponent( obj );
s = I.q .^ p;
