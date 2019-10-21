function y = evaluateDistance( obj, I, J )
%% EVALUATEDISTANCE  Evaluate locally-scaled distance 
% 
% Modified 2014/06/13


switch nargin
    case 2
        argStr = 'I.x, I.xiNorm';
    case 3
        argStr = 'I.x, I.xiNorm, J.x, J.xiNorm';
end

switch getMode( obj )
    case 'explicit'
        dStr = 'wdmat';
        eStr = '';
    case 'implicit'
        dStr = 'wldmat';
        eStr = 'I.idxE, ';
end


switch getNormalization( obj )
    case 'geometric'
        normStr = '_g';
    case 'harmonic'
        normStr = '_h';
end


eval( [ 'y = ' dStr normStr '(' eStr argStr ');' ] );


