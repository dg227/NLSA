function tag = concatenateComponentTags( obj )
% CONCATENATECOMPONENTTAGS Concatenate component tags of a matrix of nlsaEmbeddedComponent objects
%
% Modified 2014/08/07

tag = concatenateComponentTags@nlsaComponent( obj );
for iC = 1 : numel( tag )
    tag{ iC } = [ tag{ iC } '_' getEmbeddingTag( obj( iC, 1 ) ) ];
end
