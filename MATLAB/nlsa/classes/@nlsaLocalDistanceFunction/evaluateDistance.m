function y = evaluateDistance( obj, mode )
% EVALUATEDISTANCE  
%
%
% Modified 2015/10/29

lDist = getLocalDistance( obj );
if nargin == 2 && strcmp( mode, 'self' )
    % Use only query data
    y = evaluateDistance( lDist, obj.Q );  
else
    % Use both query and test data
    y = evaluateDistance( lDist, obj.Q, obj.T );
end


