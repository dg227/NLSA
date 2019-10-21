function tag = concatenateComponentTags( obj )
% CONCATENATECOMPONENTTAGS Concatenate component tags of a matrix of nlsaEmbeddedComponent objects
%
% Modified 2014/07/29

nC = size( obj, 1 );
tag  = cell( 1, nC );
for iC = 1 : nC
    tag{ iC } = getComponentTag( obj( iC, 1 ) );
end
