function tag = concatenateComponentTags( obj )
% CONCATENATECOMPONENTTAGS Concatenate component tags of a matrix of 
% nlsaEmbeddedComponent_ose objects
%
% Modified 2015/12/14

tag = concatenateComponentTags@nlsaEmbeddedComponent_xi_e( obj );
for iC = 1 : numel( tag )
    tag{ iC } = strjoin( { tag{ iC } getOseTag( obj( iC, 1 ) ) }, ...
                           '_' );
end
