function y = evaluateDistance( obj, mode )
% EVALUATEDISTANCE  
%
%
% Modified 2015/10/31

lScl  = getLocalScaling( obj );
if nargin == 2 && strcmp( mode, 'self' )
    % Use only query data
    y  = evaluateDistance@nlsaLocalDistanceFunction( obj, mode );
    sQ = evaluateScaling( lScl, obj.QS );
    y  = bsxfun( @times, sQ', y );
    y  = bsxfun( @times, y, sQ );
else
    % Use both query and test data
    y = evaluateDistance@nlsaLocalDistanceFunction( obj );
    sQ = evaluateScaling( lScl, obj.QS );
    sT = evaluateScaling( lScl, obj.QT );
    y = bsxfun( @times, sQ', y );
    y = bsxfun( @times, y, sT );
end


