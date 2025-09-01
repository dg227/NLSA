function nR = getNRealization( obj )
% GETNCOMPONENT  Get number of realizations in an nlsaComponent array
%
% Modified  2015/08/25

if ~isCompatible( obj )
    error( 'Incompatible component array' )
end
nR = size( obj, 2 );
