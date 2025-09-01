function setEigenvalues( obj, gamma, E )
% SETEIGENVALUES  Save eigenvalues of an nlsaKoopmanOperator_diff object
%
% Modified 2020/04/12

% Set eigenvalues using the parent method
setEigenvalues@nlsaKoopmanOperator( obj, gamma )

if nargin == 2 || isempty( E )
    return
end

% Set Dirichlet energies
if ~isrow( E ) || ~isnumeric( E ) 
    error( 'Dirichlet energies must be specified as a numeric row vector' )
end
if numel( E ) ~= getNEigenfunction( obj )
    msgStr = [ 'Number of Dirichlet energies must be equal to the ' ...
               'number of eigenfunctions.' ];
    error( msgStr )
end 
save( fullfile( getEigenfunctionPath( obj ), getEigenvalueFile( obj ) ), ...
      'E', '-append' )
