function a = getLinearMap( obj, idxA )
% GETLINEARMAP Get linear map of an nlsaModel object
%
% Modified 2014/02/01

if nargin == 1
    idxA = 1 : numel( obj.linMap );
end

a = obj.linMap( idxA );

