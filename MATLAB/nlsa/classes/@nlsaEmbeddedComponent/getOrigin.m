function idxO = getOrigin( obj )
% GETORIGIN Get origin property of an array of nlsaEmbeddedComponent objects
%
% Modified 2013/01/10

idxO = zeros( size( obj ) );
for iObj = 1 : numel( obj )
    idxO( iObj ) = obj( iObj ).idxO;
end
