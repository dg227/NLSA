function s = evaluateScaling( obj, I )
%% EVALUATESCALING  Evaluate shifted power law scaling 
%
% Modified 2014/06/18

[ bX, bXi, c ] = getConstant( obj );
[ pX, pXi, p ] = getExponent( obj );

s = c * ones( size( I.xiRef ) );

if bX ~= 0
    s = s + bX * ( I.x ./ I.xiRef ) .^ pX;
end
if bXi ~= 0
    s = s + bXi * ( I.xi ./ I.xiRef .^ 2 ) .^ pXi; 
end

s = s .^ p;
