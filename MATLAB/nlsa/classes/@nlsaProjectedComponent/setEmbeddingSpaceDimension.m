function obj = setEmbeddingSpaceDimension( obj, nDE )
% SETEMBEDDINGSPACEDIMENSION  Set dimension of an nlsaProjectedComponent object
%
% Modified 2014/06/23

if ~ispsi( nDE )
    error( 'Dimension must be a positive scalar integer' )
end
obj.nDE = nDE;
