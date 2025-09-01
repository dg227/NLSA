function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of an nlsaLocalScaling_exp object
%
% Modified 2014/06/18

[ bX, bXi ] = getConstant( obj );
[ pX, pXi ] = getExponent( obj );


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

tag = strcat( 'exp', tagX, tagXi ); 
