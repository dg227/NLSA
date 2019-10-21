function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaLocalScaling_pwr object
%
% Modified 2014/06/18


[ bX, bXi, c ] = getConstant( obj );
[ pX, pXi, p ] = getExponent( obj );

if bX ~= 0
    tagX = sprintf( '_bX%1.2g_pX_%1.2g', bX, pX );
else
    tagX = [];
end

if bXi ~= 0
    tagXi = sprintf( '_bXi%1.2g_pXi%1.2g', bXi, pXi );
else
    tagXi = [];
end

tag = sprintf( 'pwr_c%1.2g_p%1.2g', c, p );
tag = strcat( tag, tagX, tagXi ); 
