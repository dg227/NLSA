function computeData( obj, src, diffOp, varargin )
% COMPUTEDATA Perform out-of-sample extension of the 
% lagged embedded data of an array of nlsaEmbeddedComponent_ose objects.
%
% This mehod overloads the makeEmbedding method of the 
% nlsaEmbeddedComponent superclass
%
% Modified 2014/06/13

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
msgId = [ obj.getErrMsgId ':computeData:' ];
if ~isa( src, 'nlsaEmbeddedComponent' )
    msgStr = 'Second argument must be an nlsaEmbeddedComponent object.'; 
    error( [ msgId 'invalidSrc' ], msgStr ) 
end
if ~isCompatible( obj, src, 'testSamples', false )
    msgStr = 'Incompatible OSE and source components.';
    error( [ msgId 'incompatibleComp' ], msgStr ) 
end
if ~isa( diffOp, 'nlsaDiffusionOperator_ose' )
    msgStr = 'Third input argument must be an nlsaDiffusionOperator_ose object.'; 
    error( [ msgId 'invalidDiffOp' ], msgStr )
end
partition  = getPartition( obj( 1, : ) );
partitionT = getPartition( src( 1, : ) ); 
if any( ~isequal( partition, getPartition( diffOp ) ) )
    msgStr = 'Incompatible partitions.';
    error( [ msgId 'incompatiblePartitions' ], msgStr )
end
if any( ~isequal( partitionT, getPartitionTest( diffOp ) ) )
    msgStr = 'Incompatible test partitions';
    error( [ msgId 'incompatibleTestPartitions' ], msgStr )
end
nN = getNNeighbors( diffOp );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup global partitions for the out-of-sample and in-sample data 
% idxBG are the global indices of the batch boundaries
[ partitionG,  idxBG  ] = mergePartitions( partition );
[ partitionGT, idxBGT ] = mergePartitions( partitionT );
[ nC, nR ]              = size( obj );
nRT                     = size( src, 2 );
nBGT                    = getNBatch( partitionGT );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.batch     = 1 : getNBatch( partitionG );
Opt.logFile   = '';
Opt.logPath   = getDataPath( obj );
Opt           = parseargs( Opt, varargin{ : } );
iBatch = Opt.batch;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile, write calculation summary
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'w' );
end
clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'makeEmbedding starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of components                 = %i, \n', nC );
fprintf( logId, 'Number of in-sample realizations     = %i, \n', nR );
fprintf( logId, 'Number of out-of-sample realizations = %i, \n', nR );
fprintf( logId, 'Number of nearest neighrbors         = %i, \n', nN );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the out-of-sample realizations and batches
xOSE     = cell( nC, 1 );
for iBG = Opt.batch

    iR  = idxBG( 1, iBG ); 
    iB  = idxBG( 2, iBG );
    nB  = getNBatch( partition( iR ) );
    nSB = getBatchSize( partition( iR ), iB );
        
    % Allocate OSE data arrays
    tic
    for iC = 1 : nC
        nDE = getEmbeddingSpaceDimension( obj( iC ) );
        xOSE{ iC } = zeros( nDE, nSB );
    end
    tWall = toc;
    fprintf( logId, 'ALLOCX %2.4f \n', tWall );

    % Read OSE operator for current OSE batch
    tic
    [ p, idxG ] = getOperator( diffOp, iB, iR );
    idxG = double( idxG );
    tWall = toc;
    fprintf( logId, 'READP realizationO %i/%i, batchO %i/%i %2.4f \n', iR, nR, iB, nB, tWall );

    % Loop over the in-sample realizations and batches
    for iBGT = 1 : nBGT

        iRT  = idxBGT( 1, iBGT );
        iBT  = idxBGT( 2, iBGT );
        nBT  = getNBatch( partitionT( iRT ) );
        nSBT = getBatchSize( partitionT( iRT ), iBT );
        limT = getBatchLimit( partitionT( iRT ), iBT ); 
                
        % Form batch-local OSE operator
        tic
        ifB        = idxG >= limT( 1 ) & idxG <= limT( 2 );
        subJ       = idxG( ifB ) - limT( 1 ) + 1;
        [ subI, ~ ] = find( ifB );
        idxB       = sub2ind( [ nSB nSBT ], subI, subJ );
        pB         = zeros( nSB, nSBT );
        pB( idxB ) = p( ifB );
        tWall = toc;
        fprintf( logId, 'PB realizationO %i/%i, batchO %i/%i, realization %i/%i, batch, %i/%i, %2.4f \n', iR, nR, iB, nB, iRT, nRT, iBT, nBT, tWall );

        % Loop over the components
        for iC = 1 : nC
                
            % Read in-sample data
            tic
            x = getData( src( iC, iRT ), iBT );
            tWall = toc;
            fprintf( logId, 'READX realization %i/%i, batch %i/%i, component %i/%i, %2.4f \n', iRT, nRT, iBT, nBT, iC, nC, tWall );

            % Add contribution from current batch
            tic
            for iSB = 1 : nSB
                idxSB = find( pB( iSB, : ) ); 
                xOSE{ iC }( :, iSB ) = xOSE{ iC }( :, iSB ) ...
                                      + sum( bsxfun( @times, pB( iSB, idxSB ), x( :, idxSB ) ), 2 );
            end
            tWall = toc;
            fprintf( logId, 'ADDX realization %i/%i, batch %i/%i, component %i/%i, %2.4f \n', iRT, nRT, iBT, nBT, iC, nC, tWall );
                    
        end % component loop
    end % in-sample global batch loop
    
    % Compute norm, write OSE data
    for iC = 1 : nC
        tic 
        setData( obj( iC, iR ), xOSE{ iC }, iB )
        tWall = toc;
        fprintf( logId, 'WRITEX realizationO %i/%i, batchO %i/%i, component %i/%i $2.4f \n', iR, nR, iB, nB, tWall );
    end % component loop
end % out-of-sample global batch loop

clk = clock; % Exit gracefully
fprintf( logId, 'makeEmbedding finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
