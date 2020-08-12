function setEigenfunctions( obj, v, mu, varargin )
% SETEIGENFUNCTIONS  Set eigenfunction data of an nlsaDiffusionOperator_gl
% object
%
% Modified 2020/08/12

if ~isnumeric( v ) || ~ismatrix( v )
    error( 'Eigenfunctions must be specified as a numeric matrix' )
end

if size( v, 1 ) ~= getNTotalSample( obj  )
    error( 'Incompatible number of samples' )
end
if size( v, 2 ) ~= getNEigenfunction( obj )
    error( 'Incompatible number of eigenfunctions' )
end

if ~isnumeric( mu ) || ~iscolumn( mu )
    msgStr = [ 'Inner product weights must be specified as a numeric ' ...
               'column vector' ];
    error( msgStr )
end 
if numel( mu ) ~= size( v, 1 )
    error( 'Incompatible number of samples in Riemannian measure' )
end

file = fullfile( getEigenfunctionPath( obj ), ... 
                 getEigenfunctionFile( obj ) );
save( file, 'v', 'mu', varargin{ : } )

