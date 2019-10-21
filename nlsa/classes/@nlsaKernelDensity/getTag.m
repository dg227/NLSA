function tag = getTag( obj )
% GETTAG  Get tag of an nlsaKernelDensity object
%
% Modified 2015/10/27

if isscalar( obj )
    tag = obj.tag;
else
    tag = cell( size( obj ) );
    for iObj = 1 : numel( obj )
        tag{ i } = obj( iObj ).tag;
    end
end
