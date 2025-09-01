function file = getDefaultVelocityFile( obj )
% GETDEFAULTVELOCITYFILE Get default phase space velocity files 
% for an nlsaEmbeddedComponent_xi object
%
% Modified 2014/04/06

file = getDefaultFile( getVelocityFilelist( obj ), ...
                       getPartition( obj ), 'dataXi' );
