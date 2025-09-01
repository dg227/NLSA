function res = isequal( obj1, obj2 )
% ISEQUAL  Returns true if two nlsaPartition arrays are identical, false otherwise
%
% Modified 2014/06/09

siz1 = size( obj1 );
siz2 = size( obj2 );
if numel( siz1 ) ~= numel( siz2 )
    res = NaN;
    return
end
if any( siz1 ~= siz2 )
    res = NaN;
    return
end

res = true( size( obj1 ) );
for iObj = 1 : numel( obj1 )
    if getNBatch( obj1( iObj ) ) ~= getNBatch( obj2( iObj ) )
        res( iObj ) = false;
    end
    if res( iObj )
        res( iObj ) = all( getIdx( obj1( iObj ) ) == getIdx( obj2( iObj ) ) );
    end
end



