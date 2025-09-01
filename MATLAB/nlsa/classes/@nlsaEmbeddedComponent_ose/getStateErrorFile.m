function file = getStateErrorFile( obj, iB )
% GETSTATEERRORFILE Get state error file of an nlsaEmbeddedComponent_ose object 
%
% Modified 2014/05/20

if nargin == 1
    iB = 1 : getNBatch( obj );
end

file = getFile( getStateErrorFilelist( obj ), iB ); 

