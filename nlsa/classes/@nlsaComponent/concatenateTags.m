function tag = concatenateTags( obj )
% CONCATENATETAGS Concatenate tags of a matrix of nlsaComponent objects
%
% Modified 2014/08/07

tagC = concatenateComponentTags( obj );
tagR = concatenateRealizationTags( obj );

tag = { strjoin( tagC, '_' ) ...
        strjoin( tagR, '_' ) };

