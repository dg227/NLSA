function obj = setOrigin( obj, idxO )
% SETORIGIN Set origin property of nlsaEmbeddedComponent object
%
% Modified 2012/12/20

if ~ispsi( idxO )
    error( 'Time origin must be a positive scalar integer' )
end
obj.idxO = idxO;

