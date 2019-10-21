function tag = getDefaultEmbeddingTag( obj )
% GETDEFAULTEMBEDDINGTAG  Get default embedding tag of nlsaEmbeddedComponent 
% objects
%
% Modified 2014/07/29

idxE      = getEmbeddingIndices( obj );
idxO      = getOrigin( obj );
idxT      = idxO - 1 + [ 1 getNSample( obj ) ];  

tag = idx2str( idxE, 'idxE' );

tag = [ tag '_idxT' int2str( idxT( 1 ) ), ...
            '-'     int2str( idxT( end ) ) ];
