function y = evaluateDistance( obj, I, J )
%% EVALUATEDISTANCE  Evaluate cone distance 
% 
% Modified 2015/09/11

alpha = getAlpha( obj );
zeta  = getZeta( obj );
tol   = getTolerance( obj );

switch nargin
    case 2
        argStr = 'zeta, tol, I.x, I.xi, I.xiNorm';
    case 3
        argStr = 'zeta, tol, I.x, I.xi, I.xiNorm, J.x, J.xi, J.xiNorm';
end

switch getMode( obj ) 
    case 'explicit'
        dStr = 'cdmat2';
        eStr = '';
    case 'implicit'
        dStr = 'cldmat2';
        eStr = 'I.idxE, ';
end

switch getNormalization( obj )
    case 'geometric'
        normStr = '_g';
    case 'harmonic'
        normStr = '_h';
end

eval( [ 'y = ' dStr normStr '(' eStr, argStr ');' ] );

if alpha ~= 0
    switch nargin
        case 2
            y = bsxfun( @times, ( I.xiNorm.^-alpha )', y );
            y = bsxfun( @times, y, ( I.xiNorm.^-alpha )' );
        case 3
            y = bsxfun( @times, ( I.xiNorm.^-alpha )', y );
            y = bsxfun( @times, y, ( J.xiNorm.^-alpha ) );
    end
end
