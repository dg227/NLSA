function pth = getLeftSingularVectorPath( obj )
% GETLEFTSINGULARVECTORPATH  Get the left singular vector path of an 
% nlsaCovarianceOperator object 
%
% Modified 2014/07/16

pth = fullfile( getPath( obj ), getLeftSingularVectorSubpath( obj ) );
