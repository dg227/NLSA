function tag = getDefaultTag( obj )
% GETDEFAULTTAG  Get default tag of nlsaLocalDistance_sone object
%
% Modified 2015/03/30

if ifVNorm( obj )
    tagV = '_vNorm';
else
    tagV = '';
end

tag = [ 'sone_zeta' num2str( getZeta( obj ) ) ...
        '_tol' num2str( getTolerance( obj ), '%1.2E' ) ...
        tagV, ...
        '_' getNormalization( obj ) ];
