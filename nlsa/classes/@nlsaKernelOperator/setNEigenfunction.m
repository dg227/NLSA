function obj = setNEigenfunction( obj, nEig )
% SETNNEIGENFUNCTION  Set number of eigenfunctions in an nlsaKernelOperator
% object
%
% Modified 2014/07/16

if ~ispsi( nEig )
    error( 'The number of eigenfunctions must be a positive scalar integer' )
end
obj.nEig = nEig;
