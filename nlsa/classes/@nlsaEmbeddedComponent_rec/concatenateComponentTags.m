function tag = concatenateComponentTags( obj )
% CONCATENATECOMPONENTTAGS Concatenate component tags of a matrix of 
% nlsaEmbeddedComponent_rec objects
%
% Modified 2014/08/04

tag = concatenateComponentTags@nlsaEmbeddedComponent_xi_e( obj );
for iC = 1 : numel( tag )
    tag{ iC } = strjoin( tag{ iC }, getBasisFunctionTag( obj( iC, 1 ) ), '_' );
end
