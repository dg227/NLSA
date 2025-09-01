function obj = setVelocityTag( obj, tag )
% SETVELOCITYTAG Set velocity tag property of an nlsaEmbeddedComponent_xi object
%
% Modified 2014/08/07

if ~( isrowstr( tag )  )
    error( 'Invalid tag specification' )
end

obj.tagXi = tag;
