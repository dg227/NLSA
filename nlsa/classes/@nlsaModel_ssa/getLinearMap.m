function a = getLinearMap( obj, idxA )
% GETLINEARMAP Get linear map of an nlsaModel_ssa object
%
% Modified 2016/05/31

if nargin == 1
    idxA = 1 : numel( obj.linMap );
end

a = obj.linMap( idxA );

