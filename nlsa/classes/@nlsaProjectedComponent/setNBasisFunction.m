function obj = setNBasisFunction( obj, nL )
% SETNBASISFUNCTION Set the number of basis functions of an 
% nlsaProjectedComponent object
%
% Modified 2014/06/20

if ~ispsi( nL )
    error( 'Number of basis functions must be a positive scalar integer' )
end

obj.nL = nL;
