function nDE = getEmbeddingSpaceDimension( obj, iC )
% GETEMBEDDINGSPACEDIMENSION  Get embedding space dimension of an
% array of nlsaProjectedComponent objects
%
% Modified 2015/09/22

if nargin == 1
    nDE = zeros( size( obj ) );
    for iObj = 1 : numel( obj )
        nDE( iObj ) = obj( iObj ).nDE; 
    end
else
    nObj = numel( iC );
    nDE = zeros( 1, nObj );
    for iObj = 1 : nObj
        nDE( iObj ) = obj( iC( iObj ) ).nDE;
    end
end
