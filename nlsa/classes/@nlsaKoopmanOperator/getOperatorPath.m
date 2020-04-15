function path = getOperatorPath( obj )
% GETOPERATORPATH  Get operator path of an nlsaKoopmanOperator object
%
% Modified 2020/04/15

path = fullfile( getPath( obj ), getOperatorSubpath( obj ) );
