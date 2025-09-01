function obj = setTag( obj, tag )
% SETTAG  Set path of nlsaPairwiseDistance object
%
% Modified 2014/04/10


if ~isrowstr( tag ) && ~iscell( tag )
    error( 'nlsaPairwiseDistance:setTag:invalidTag', ...
           'Tag property must be a character string or a cell array' )
end
obj.tag = tag;
