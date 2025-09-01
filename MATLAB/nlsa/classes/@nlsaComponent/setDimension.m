function obj = setDimension( obj, nD )
% SETDIMENSION  Set dimension of nlsaComponent object
%
% Modified 2012/12/20

if ~ispsi( nD )
    error( 'Dimension must be a positive scalar integer' )
end
obj.nD = nD;
