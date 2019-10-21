function pth = getDefaultDistanceSubpath( obj )
% GETDEFAULTDDISTANCESUBPATH Get default distance subpath of 
% an nlsaSymmetricDistance_batch object
%
% Modified 2014/04/30

pth = sprintf( 'dataYS_nNMax%i', getNNeighborsMax( obj ) );
