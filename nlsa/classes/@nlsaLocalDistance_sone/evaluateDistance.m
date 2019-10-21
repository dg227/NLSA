function y = evaluateDistance( obj, I, J )
%% EVALUATEDISTANCE  Evaluate sine cone distance 
% 
% Modified 2015/03/30


zeta = getZeta( obj );
tol  = getTolerance( obj );

switch nargin
    case 2
        argStr = 'zeta, tol, I.x, I.xi, I.xiNorm';
    case 3
        argStr = 'zeta, tol, I.x, I.xi, I.xiNorm, J.x, J.xi, J.xiNorm';
end

switch getMode( obj ) 
    case 'explicit'
        dStr = 'sdmat';
        eStr = '';
    case 'implicit'
        dStr = 'sldmat';
        eStr = 'I.idxE, ';
end

if ifVNorm( obj )
    vStr = '';
else
    vStr = '2';
end

switch getNormalization( obj )
    case 'geometric'
        normStr = '_g';
    case 'harmonic'
        normStr = '_h';
end

eval( [ 'y = ' dStr vStr normStr '(' eStr, argStr ');' ] );

