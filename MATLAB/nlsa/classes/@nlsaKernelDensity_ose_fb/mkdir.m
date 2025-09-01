function mkdir( obj ) 
% MKDIR Make directories of nlsaKernelDensity_fb objects
%
% Modified 2018/07/05

for iObj = 1 : numel( obj )
    pth = getDensityPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
