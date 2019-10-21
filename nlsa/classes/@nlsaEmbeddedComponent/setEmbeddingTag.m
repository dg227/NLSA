function obj = setEmbeddingTag( obj, tag )
% SETEMBEDDINGTAG Set embedding tag property of an nlsaEmbeddedComponent object
%
% Modified 2014/07/29

if ~( isrowstr( tag )  )
    error( 'Invalid tag specification' )
end

obj.tagE = tag;
