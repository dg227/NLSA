function y = evaluateDistance( obj, I, J )
% EVALDIST  Evaluate L2 distance scaled at each Takens delay 
%
% Modified 2015/11/03

lScl = getLocalScaling( obj );
mode = getMode( obj );

nE = numel( I.idxE );
nDE = size( I.x, 1 );
nD  = nDE / nE;


switch mode
    case 'explicit'
        nSI = size( I.x, 2 );
    case 'implicit'
        nSI = size( I.x, 2 ) - nE + 1;
end
sI = evaluateScaling( lScl, I.S );
if nargin == 3
    nSJ = size( J.x, 2 );
    sJ  = evaluateScaling( lScl, J.S );
end

switch mode
    case 'explicit'
        switch nargin
            case 2
                y = zeros( nSI );
                for iE =  1 : nE
                    iDE1 = ( iE - 1 ) * nD + 1;
                    iDE2 = iE * nD;
                    yAdd = dmat( I.x( iDE1 : iDE2, :  ) );
                    yAdd = bsxfun( @times, sI( iE, : )', yAdd );
                    yAdd = bsxfun( @times, yAdd, sI( iE, : ) );
                    y    = y + yAdd;
                end
            case 3
                y   = zeros( nSI, nSJ );
                for iE = 1 : nE
                    iDE1 = ( iE - 1 ) * nD + 1;
                    iDE2 = iE * nD;
                    yAdd = dmat( I.x( iDE1 : iDE2, : ), ...
                                 J.x( iDE1 : iDE2, : ) );
                    yAdd = bsxfun( @times, sI( iE, : )', yAdd );
                    yAdd = bsxfun( @times, yAdd, sJ( iE, : ) );
                    y    = y + yAdd;
                end
        end
    case 'implicit'
        switch nargin
            case 2
                y = ldmat_scl( I.idxE, I.x, sI );
            case 3
                y = ldmat_scl( I.idxE, I.x, sI, J.x, sJ );
    end
end
