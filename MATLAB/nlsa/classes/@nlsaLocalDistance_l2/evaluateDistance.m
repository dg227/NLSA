function y = evaluateDistance( obj, I, J )
% EVALDIST  Evaluate L2 distance 
%
% Modified 2020/07/11


switch nargin
case 2
    switch getMode( obj )
    case 'explicit'
        y = dmat( I.x );
    case 'implicit'
        y = ldmat_par( I.idxE, I.x, [], obj.nPar );
    end
case 3
    switch getMode( obj );
    case 'explicit'
        y = dmat( I.x, J.x );
    case 'implicit'
        y = ldmat_par( I.idxE, I.x, J.x, obj.nPar );
    end
end

