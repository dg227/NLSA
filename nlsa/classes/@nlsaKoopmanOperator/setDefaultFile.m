function obj = setDefaultFile( obj )
% SETDEFAULTFILE Set default filenames for an nlsaKoopmanOperator object 
%
% Modified 2020/04/11

obj = setOperatorFile( obj, getDefaultOperatorFile( obj ) );
obj = setEigenvalueFile( obj, getDefaultEigenvalueFile( obj ) );
obj = setEigenfunctionFile( obj, getDefaultEigenfunctionFile( obj ) );
obj = setEigenfunctionCoefficientFile( obj, ...
        getDefauleEigenfunctionCoefficientFile( obj ) );
