function nC = getNSrcComponent( obj )
% GETNSRCCOMPONENT Get number of source components of nlsaModel_base objects
%
% Modified 2013/10/15

nC = size( obj.srcComponent, 1 );
