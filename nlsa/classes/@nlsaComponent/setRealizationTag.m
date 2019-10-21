function obj = setRealizationTag( obj, tag )
% SETREALIZATIONTAG Set realization tag property of nlsaComponent object
%
% Modified 2014/07/28

if ~( isrowstr( tag )  )
    error( 'Invalid tag specification' )
end

obj.tagR = tag;
