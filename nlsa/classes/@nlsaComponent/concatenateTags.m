function tag = concatenateTags( obj )
% CONCATENATETAGS Concatenate tags of a matrix of nlsaComponent objects
%
% Modified 2019/11/05

tag = { concatenateComponentTags( obj ) concatenateRealizationTags( obj ) };

