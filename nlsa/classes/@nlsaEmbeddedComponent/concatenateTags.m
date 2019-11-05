function tag = concatenateTags( obj )
% CONCATENATETAGS Concatenate tags of an array of nlsaEmbeddedComponent objects
%
% Modified 2019/11/05

tagC = concatenateComponentTags( obj );
tagR = concatenateRealizationTags( obj );
tagE = concatenateEmbeddingTags( obj )l

tag = { strjoin_e( tagC, '_' ) ...
        strjoin_e( tagR, '_' ) ...
        strjoin_e( tagE, '_' ) };

