function obj = setNNeighbors( obj, nN )
% SETNNEIGHBORS  Set number of nearest neighrbors of an nlsaSymmetricDistance 
% object
%
% Modified 2012/12/21

if ispsi( nN )
    obj.nN = nN;
else
    error( 'Number of nearest neighbors must be a positive scalar integer' )
end
