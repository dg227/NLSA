function pth = getDefaultProjectionSubpath( obj )
% GETDEFAULTPROJECTIOnSUBPATH Get default projection subpath of 
% an nlsaProjectedComponent object
%
% Modified 2014/06/23

pth = strcat( 'dataA_', getDefaultTag( obj ) );
