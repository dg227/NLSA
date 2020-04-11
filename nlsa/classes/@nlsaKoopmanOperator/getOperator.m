function V = getOperator( obj )
% GETOPERATOR  Read operator data of an nlsaKoopmanOperator object
%
% Modified 2020/04/10


varNames = { 'V' };
file = fullfile( getOperatorPath( obj ), getOperatorFile( obj ) );
load( file, varNames{ 1 : nargout } )
