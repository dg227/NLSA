function mkdir( obj ) 
% MKDIR Make directories of nlsaKernelDensity_fb objects
%
% Modified 2015/04/08

for iObj = 1 : numel( obj )
    pth = getDensityPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
