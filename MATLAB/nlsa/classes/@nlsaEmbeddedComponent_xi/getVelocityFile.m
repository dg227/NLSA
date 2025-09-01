function file = getVelocityFile( obj, iB )
% GETVELOCITYFILE Get phase space velocity file of an nlsaEmbeddedComponent_xi 
% object 
%
% Modified 2014/04/06

if nargin == 1
    iB = 1 : getNBatch( obj );
end

file = getFile( getVelocityFilelist( obj ), iB ); 

