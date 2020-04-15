function obj = setDefaultSubpath( obj )
% SETDEFAULTSUBPATH Set default subdirectories of an nlsaKoopmanOperator 
% object 
%
% Modified 2020/04/15

obj = setOperatorSubpath( obj, getDefaultOperatorSubpath( obj ) );
obj = setEigenfunctionSubpath( obj, getDefaultEigenfunctionSubpath( obj ) );
