function w = getFDWeights( obj )
% GETFDWEIGHTS Get finite difference weights of nlsaKoopman operator object
%
% Modified 2020/04/17

dt = getSamplingInterval( obj );
w = fdWeights( getFDOrder( obj ), getFDType( obj ) ) / dt;
