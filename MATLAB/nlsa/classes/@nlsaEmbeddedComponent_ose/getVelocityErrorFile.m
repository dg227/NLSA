function file = getVelocityErrorFile( obj, iB )
% GETVELOCITYERRORFILE Get phase space velocity error file of 
% nlsaEmbeddedComponent_ose object 
%
% Modified 2014/04/06

if nargin == 1
    iB = 1 : getNBatch( obj );
end

file = getFile( getVelocityErrorFilelist( obj ), iB ); 

