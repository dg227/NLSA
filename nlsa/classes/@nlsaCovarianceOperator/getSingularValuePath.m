function pth = getSingularValuePath( obj )
% GETSINGULARVALUEPATH  Get singular value path of an 
% nlsaCovarianceOperator object 
%
% Modified 2014/08/10

pth = fullfile( getPath( obj ), getSingularValueSubpath( obj ) );
