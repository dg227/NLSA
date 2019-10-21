function pth = getDataPath( obj )
% GETDATAPATH Get data path of an array of nlsaComponent objects
%
% Modified 2015/08/31

pth = getPath( obj );

if isscalar( obj )
    pth = fullfile( pth, getDataSubpath( obj ) );
else
    for iObj = 1 : numel( obj )
        pth{ iObj } = fullfile( pth{ iObj }, getDataSubpath( obj( iObj ) ) ) ;
    end
end
