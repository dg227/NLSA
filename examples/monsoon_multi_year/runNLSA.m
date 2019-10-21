experiment = 'std';
iProc = 1;
nProc = 1;
disp( experiment )
model = monsoonNLSAModel( experiment ); 


%disp( 'Embedding' ); makeEmbedding( model )

% Only needed for cone kernel
%disp( 'Phase space velocity' ); computeVelocity( model )

% Not needed if source and target data are the same
%disp( 'Target embedding' ); makeTrgEmbedding( model );
%disp( 'Target velocity' ); computeTrgVelocity( model, 'ifWriteXi', true )


%fprintf( 'Pairwise distances, %i/%i\n', iProc, nProc ); computePairwiseDistances( model, iProc, nProc )
%disp( 'Distance symmetrization \n' ); symmetrizeDistances( model )

%fprintf( 'Kernel double sum \n' ); computeKernelDoubleSum( model );


%disp( 'Diffusion eigenfunctions' ); computeDiffusionEigenfunctions( model )
%phi = getDiffusionEigenfunctions( model ); % returns the eigenfunctions

%disp( 'Projection' ); computeProjection( model );
%a = getProjectedData( model ); % returns the target data projected onto the eigenfunctions


