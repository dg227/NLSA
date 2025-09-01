function obj = setBasisFunctionTag( obj, tag )
% SETBASISFUNCTIONTAG Set basis function tag property of an 
% nlsaComponent_rec object
%
% Modified 2015/09/11

if ~( isrowstr( tag )  )
    error( 'Invalid tag specification' )
end

obj.tagPhi = tag;
