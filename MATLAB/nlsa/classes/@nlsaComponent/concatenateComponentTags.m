function tag = concatenateComponentTags( obj )
% CONCATENATECOMPONENTTAGS Concatenate component tags of a matrix of 
% nlsaComponent objects
%
% Modified 2019/11/05

nC = size( obj, 1 );
tag  = cell( 1, nC );
for iC = 1 : nC
    tag{ iC } = getComponentTag( obj( iC, 1 ) );
end
tag = strjoin_e( tag, '_' );
