function tag = concatenateTags( obj )
% CONCATENATETAGS Concatenate tags of a matrix of nlsaEmbeddedComponent objects
%
% Modified 2019/11/05

tag = [ concatenateTags@nlsaComponent( obj ) ...
        { concatenateEmbeddingTags( obj ) } ];

