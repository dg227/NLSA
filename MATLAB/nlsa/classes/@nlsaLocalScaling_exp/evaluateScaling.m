function s = evaluateScaling( obj, I )
%% EVALUATESCALING  Evaluate exponential scaling 
%
% Modified 2014/06/18

[ bX, bXi ] = getConstant( obj );
[ pX, pXi ] = getExponent( obj );

s = zeros( size( I.xiRef ) );

if bX ~= 0
    s = s + bX * ( I.x ./ I.xiRef ) .^ pX;
end
if bXi ~= 0
    s = s + bXi * ( I.xi ./ I.xiRef .^ 2 ) .^ pXi; 
end

s = exp( s );
