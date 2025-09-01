function w = getFDWeights( obj )
% GETFDWEIGHTS Get finite difference weights of nlsaEmbeddedComponent_xi object
%
% Modified 2020/04/15

w = fdWeights( getFDOrder( obj ), getFDType( obj ) );
