function nC = getNComponent( obj )
% GETNCOMPONENT  Get number of components in an nlsaComponent array
%
% Modified  2015/08/25

if ~isCompatible( obj )
    error( 'Incompatible component array' )
end
nC = size( obj, 1 );
