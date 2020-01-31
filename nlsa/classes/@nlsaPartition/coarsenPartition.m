function [ objC, idxFC ] = coarsenPartition( objF, prt );
% COARSENPARTITION Coarsen an nlsaPartition object objF by merging the 
% batches specified by a coarsening partition prt
%
% Modified 2019/11/16

% Validate first input argument 
if ~isscalar( objF )
    errror( 'First argument must be a scalar nlsaPartition object' )
end

% Create a coarsening partition if prt is a positive scalar integer
if ispsi( prt ) 
    prt = nlsaPartition( 'nSample', getNBatch( objF ), 'nBatch', prt );  
end

% Validate second input argument
if ~isa( prt, 'nlsaPartition' ) || ~isscalar( prt ) 
   error( 'Second argument must be a positive scalar integer or scalar nlsaPartition object' )  
end

idxF = getIdx( objF );         % batch indices for fine partition 
idxC = idxF( getIdx( prt ) );  % batch indices for coarse partition

objC  = nlsaPartition( 'idx', idxC ); % coarse partition 
idxFC = findBatch( objC, idxF ); % affilition of batches of fine partition in coarse partition

