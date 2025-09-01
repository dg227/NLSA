function pth = getDataSubpath( obj )
% GETDATASUBPATH Get data path of an array of nlsaComponent objects
%
% Modified 2015/08/31

pth = cell( size( obj ) );
for iObj = 1 : numel( obj );
    pth{ iObj } = obj( iObj ).pathX;
end
            
if isscalar( obj )
    pth = pth{ 1 };
end
