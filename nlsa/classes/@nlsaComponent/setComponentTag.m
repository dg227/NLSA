function obj = setComponentTag( obj, tag )
% SETCOMPONENTTAG Set component tag property of nlsaComponent object
%
% Modified 2014/07/28

if ~( isrowstr( tag )  )
    error( 'Invalid tag specification' )
end

obj.tagC = tag;
