function tag = getTag( obj )
% GETTAG  Get tag of an nlsaPairwiseDistance object
%
% Modified 2019/06/26

if isscalar( obj )
    tag = obj.tag;
else
    tag = cell( size( obj ) );
    for iObj = 1 : numel( obj )
        tag{ iObj } = obj( iObj ).tag;
    end
end
