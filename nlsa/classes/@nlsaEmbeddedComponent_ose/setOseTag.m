function obj = setOseTag( obj, tag )
% SETOSETAG Set ose tag property of an nlsaEmbeddedComponent_ose_n object
%
% Modified 2015/12/14

if ~( isrowstr( tag )  )
    error( 'Invalid tag specification' )
end

obj.tagO = tag;
