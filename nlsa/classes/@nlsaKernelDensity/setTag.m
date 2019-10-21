function obj = setTag( obj, tag )
% SETTAG  Set tag of an nlsaKernelDensity object
%
% Modified 2015/04/06

if ~isrowstr( tag ) && ~iscell( tag )
    error( 'Tag property must be a character string or a cell array' )
end
obj.tag = tag;
