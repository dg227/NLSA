function computeData( obj, src, diffOp, varargin )
% COMPUTEDATA Perform reconstruction of the lagged embedded data of an 
% array of nlsaEmbeddedComponent_rec objects.
%
% This mehod overloads the computeData method of the 
% nlsaEmbeddedComponent_ose superclass
%
% Modified 2014/07/06

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
msgId = [ obj.getErrMsgId ':computeData:' ];
if ~isa( src, 'nlsaProjectedComponent' )
    msgStr = 'Second argument must be an nlsaProjectedComponent object.'; 
    error( [ msgId 'invalidSrc' ], msgStr ) 
end
if ~isa( diffOp, 'nlsaDiffusionOperator' )
    msgStr = 'Third input argument must be an nlsaDiffusionOperator object.'; 
    error( [ msgId 'invalidDiffOp' ], msgStr )
end
partition  = getPartition( obj( 1, : ) );


if any( ~isequal( partition, getPartition( diffOp ) ) )
    msgStr = 'Incompatible partitions.';
    error( [ msgId 'incompatiblePartitions' ], msgStr )
end
nDE = getEmbeddingSpaceDimension( obj( :, 1 ) );
if any( nDE ~= getEmbeddingSpaceDimension( src ) )
    error( 'Incompatible embedding dimensions' )
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup global partitions for the out-of-sample and in-sample data 
% idxBG are the global indices of the batch boundaries
[ partitionG,  idxBG  ] = mergePartitions( partition );
[ nC, nR ]              = size( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.batch     = 1 : getNBatch( partitionG );
Opt.logFile   = '';
Opt.logPath   = getDataPath( obj );
Opt           = parseargs( Opt, varargin{ : } );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile, write calculation summary
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'w' );
end
clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeData starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of components   = %i, \n', nC );
fprintf( logId, 'Number of realizations = %i, \n', nR );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the out-of-sample realizations and batches
for iBG = Opt.batch

    iR  = idxBG( 1, iBG ); 
    iB  = idxBG( 2, iBG );
    nB  = getNBatch( partition( iR ) );
        
    % Read OSE eigenfunctions for current batch, including the constant eigenfunction
    tic
    phi = getEigenfunctions( diffOp, true, iB, iR )';
    tWall = toc;
    fprintf( logId, 'READPHI realizationO %i/%i, batchO %i/%i %2.4f \n', iR, nR, iB, nB, tWall );

    % Loop over the out-of-sample components
    for iC = 1 : nC
        tic
        idxPhi = getEigenfunctionIndices( obj( iC ) );

        x = getProjectedData( src, iC, idxPhi ) * phi( idxPhi + 1, : );
        tWall = toc;
        fprintf( logId, 'OSEX component %i/%i, realization %i/%i, batch %i/%i %2.4f \n', iC, nC, iR, nR, iB, nB, tWall );

        tic 
        setData( obj( iC, iR ), x, iB )
        tWall = toc;
        fprintf( logId, 'WRITEX component %i/%i, realization %i/%i, batch %i/%i $2.4f \n', iC, nC, iR, nR, iB, nB, tWall );
    end % component loop
end % out-of-sample global batch loop

clk = clock; % Exit gracefully
fprintf( logId, 'computeData finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
