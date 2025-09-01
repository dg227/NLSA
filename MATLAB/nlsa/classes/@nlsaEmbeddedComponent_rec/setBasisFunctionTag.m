function obj = setBasisFunctionTag( obj, tag )
% SETBASISFUNCTIONTAG Set basis function tag property of an 
% nlsaEmbeddedComponent_rec object
%
% Modified 2014/08/04

if ~( isrowstr( tag )  )
    error( 'Invalid tag specification' )
end

obj.tagPhi = tag;
