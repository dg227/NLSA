function obj = setRealizationTag( obj, tag )
% SETREALIZATIONTAG Set realization tag property of nlsaComponent objects
%
% Modified 2019/11/16

nObj = numel( obj );
if ischar( tag )
    tag = repmat( { tag }, nObj );
else ~iscellstr( tag ) || ~isequal( size( obj ), size( tag ) )
    error( 'Second input argument must be either a string or a cell array of strings of equal size to the size of the first input argument' )
end

for iObj = 1 : nObj
    obj( iObj ).tagR = tag{ iObj };
end

