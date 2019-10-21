function nF = getNFile( obj )
% GETNFILE  Get number of files in an array of nlsaFilelist objects
%
% Modified  2014/04/04

nF = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    nF( iObj ) = obj( iObj ).nF;
end
