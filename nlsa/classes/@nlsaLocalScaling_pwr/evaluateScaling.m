function s = evaluateScaling(obj, I)
%% EVALUATESCALING  Evaluate density power law scaling 
%
% Modified 2022/11/06

s = I.q .^ I.p;
