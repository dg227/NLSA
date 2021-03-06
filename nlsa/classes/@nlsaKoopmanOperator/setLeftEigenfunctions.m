function setLeftEigenfunctions( obj, zeta, mu, varargin )
% SETLEFTEIGENFUNCTIONS  Set left eigenfunction data of an nlsaKoopmanOperator
% object
%
% Modified 2020/08/27

if ~isnumeric( zeta ) || ~ismatrix( zeta )
    error( 'Eigenfunctions must be specified as a numeric matrix' )
end

if size( zeta, 1 ) ~= getNTotalSample( obj  )
    error( 'Incompatible number of samples in eigenfunction array' )
end
if size( zeta, 2 ) ~= getNEigenfunction( obj )
    error( 'Incompatible number of eigenfunctions' )
end

if ~isnumeric( mu ) || ~iscolumn( mu )
    msgStr = [ 'Inner product weights must be specified as a numeric ' ...
               'column vector' ];
    error( msgStr )
end 
if numel( mu ) ~= size( zeta, 1 )
    error( 'Incompatible number of samples in Riemannian measure' )
end

file = fullfile( getEigenfunctionPath( obj ), ... 
                 getLeftEigenfunctionFile( obj ) );
save( file, 'zeta', 'mu', varargin{ : } )

