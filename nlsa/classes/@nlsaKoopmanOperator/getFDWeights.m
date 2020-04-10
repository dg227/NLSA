function w = getFDWeights( obj )
% GETFDWEIGHTS Get finite difference weights of nlsaKoopman operator object
%
% Modified 2020/04/09

w = obj.fdWeights( getFDOrder( obj ), getFDType( obj ) );
