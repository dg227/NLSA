function y = evaluateDistance( obj, I, J )
% EVALDIST  Evaluate L2 distance 
%
% Modified 2019/10/20


switch nargin
case 2
    switch getMode( obj )
    case 'explicit'
        y = dmat( I.x );
    case 'implicit'
        y = ldmat( I.idxE, I.x );
    end
case 3
    switch getMode( obj );
    case 'explicit'
        y = dmat( I.x, J.x );
    case 'implicit'
        y = ldmat( I.idxE, I.x, J.x );
    end
end


