function pth = getRightSingularVectorPath( obj )
% GETRIGHTSINGULARVECTORPATH  Get the right singular vector path of an 
% nlsaCovarianceOperator object 
%
% Modified 2014/07/16

pth = fullfile( getPath( obj ), getRightSingularVectorSubpath( obj ) );
