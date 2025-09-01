function obj = setTag( obj, tag )
% SETTAG  Set tag of an nlsaProjectedComponent object
%
% Modified 2014/06/20

if ~isrowstr( tag ) && ~iscell( tag )
    error( 'Tag property must be a character string or a cell array' )
end
obj.tag = tag;
