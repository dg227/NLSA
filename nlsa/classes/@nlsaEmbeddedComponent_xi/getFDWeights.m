function w = getFDWeights( obj )
% GETFDWEIGHTS Get finite difference weights of nlsaEmbeddedComponent_xi object
% Modified 2014/04/14


w = obj.fdWeights( getFDOrder( obj ), getFDType( obj ) );
