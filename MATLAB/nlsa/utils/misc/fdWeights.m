function w = fdWeights( nOrd, fdType )
% FDWEIGHTS Finite difference weights for first derivative
%
% Modified 2021/06/17

switch fdType
    case 'central'
        switch nOrd
            case 2
                w = [ -1/2 0 1/2 ];
            case 4
                w = [ 1/12 -2/3 0 2/3 -1/12 ];
            case 6
                w = [ -1/60 3/20 -3/4 0 3/4 -3/20 1/60 ];
            case 8
                w = [ 1/280 -4/105 1/5 -4/5 0 4/5 -1/5 4/105 -1/280 ];  
        end
    case 'forward'
        switch nOrd
            case 1
                w = [ -1 1 ];
            case 2
                w = [ -3/2 2 -1/2 ];
            case 3
                w = [ -11/6 3 -3/2 1/3 ];
            case 4
                w = [ -25/12 4 -3 4/3 -1/4 ];
        end
    case 'backward'
        switch nOrd
            case 1
                w = [ -1 1 ];
            case 2
                w = [ 1/2 -2 3/2 ];
            case 3
                w = [ -1/3 3/2 -3 11/6 ];
            case 4
                w = [ 1/4 -4/3 3 -4 25/12 ];
        end
end
