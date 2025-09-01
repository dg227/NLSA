function pth = getOperatorPath( obj )
% GETOPERATORPATH  Get operator path of an nlsaCovarianceOperator object 
%
% Modified 2014/07/16

pth = fullfile( getPath( obj ), getOperatorSubpath( obj ) );
