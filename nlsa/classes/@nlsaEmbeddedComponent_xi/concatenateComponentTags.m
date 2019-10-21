function tag = concatenateComponentTags( obj )
% CONCATENATECOMPONENTTAGS Concatenate component tags of a matrix of 
% nlsaEmbeddedComponent_xi objects
%
% Modified 2014/08/07

tag = concatenateComponentTags@nlsaEmbeddedComponent( obj );
for iC = 1 : numel( tag )
    tag{ iC } = [ tag{ iC } '_' getVelocityTag( obj( iC, 1 ) ) ];
end
