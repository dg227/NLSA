function nC = getNTrgComponent( obj )
% GETNTRGCOMPONENT Get number of target components of nlsaModel_base objects
%
% Modified 2013/10/15

nC = size( obj.trgComponent, 1 );
