function nC = getNDenComponent( obj )
% GETNDENCOMPONENT Get number of density components of nlsaModel_den objects
%
% Modified 2014/12/30

nC = size( obj.denComponent, 1 );
