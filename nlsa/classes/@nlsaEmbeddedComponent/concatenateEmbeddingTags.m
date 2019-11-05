function tag = concatenateEmbeddingTags( obj )
% CONCATENATEEMBEDDINGTAGS Concatenate component tags of a matrix of 
% nlsaEmbeddedComponent objects
%
% Modified 2019/11/05

nC = size( obj, 1 );
tag  = cell( 1, nC );
for iC = 1 : nC
    tag{ iC } = getEmbeddingTag( obj( iC, 1 ) );
end
tag = strjoin_e( tag, '_' );
