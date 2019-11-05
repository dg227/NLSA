function tag = concatenateTags( obj )
% CONCATENATETAGS Concatenate tags of a matrix of nlsaEmbeddedComponent_xi 
% objects
%
% Modified 2019/11/05

tag = [ concatenateTags@nlsaEmbeddedComponent( obj ) ...
        { concatenateVelocityTags( obj ) } ];

