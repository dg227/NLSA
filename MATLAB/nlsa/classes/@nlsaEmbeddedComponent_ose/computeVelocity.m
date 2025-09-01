function computeVelocity( obj, src, diffOp, varargin )
% COMPUTEVELOCITY Perform out-of-sample extension of the 
% phase space velocity of an array of nlsaEmbeddedComponent_ose objects.
%
% This mehod overloads the computeVelocity method of the 
% nlsaEmbeddedComponent_xi superclass
%
% Modified 2014/06/13

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
msgId = [ obj.getErrMsgId ':computeVelocity:' ];
if ~isa( src, 'nlsaEmbeddedComponent_xi' )
    msgStr = 'Second argument must be an nlsaEmbeddedComponent_xi object.'; 
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
    msgStr = 'Incompatible partitions.' 
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
Opt.logPath   = getVelocityPath( obj );
Opt.ifWriteXi = true;
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
fprintf( logId, 'computeVelocity starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of components                 = %i, \n', nC );
fprintf( logId, 'Number of in-sample realizations     = %i, \n', nR );
fprintf( logId, 'Number of out-of-sample realizations = %i, \n', nR );
fprintf( logId, 'Number of nearest neighrbors         = %i, \n', nN );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the out-of-sample realizations and batches
xiOSE     = cell( nC, 1 );
for iBG = Opt.batch

    iR  = idxBG( 1, iBG ); 
    iB  = idxBG( 2, iBG );
    nB  = getNBatch( partition( iR ) );
    nSB = getBatchSize( partition( iR ), iB );
        
    % Allocate OSE phase-space velocity arrays
    tic
    for iC = 1 : nC
        nDE = getEmbeddingSpaceDimension( obj( iC ) );
        xiOSE{ iC } = zeros( nDE, nSB );
    end
    tWall = toc;
    fprintf( logId, 'ALLOCXI %2.4f \n', tWall );

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
                
            % Read in-sample phase space velocity
            tic
            [ ~, xi ] = getVelocity( src( iC, iRT ), iBT );
            tWall = toc;
            fprintf( logId, 'READXI realization %i/%i, batch %i/%i, component %i/%i, %2.4f \n', iRT, nRT, iBT, nBT, iC, nC, tWall );

            % Add contribution from current batch
            tic
            for iSB = 1 : nSB
                idxSB = find( pB( iSB, : ) ); 
                xiOSE{ iC }( :, iSB ) = xiOSE{ iC }( :, iSB ) ...
                                      + sum( bsxfun( @times, pB( iSB, idxSB ), xi( :, idxSB ) ), 2 );
            end
            tWall = toc;
            fprintf( logId, 'ADDXI realization %i/%i, batch %i/%i, component %i/%i, %2.4f \n', iRT, nRT, iBT, nBT, iC, nC, tWall );
                    
        end % component loop
    end % in-sample global batch loop
    
    % Compute norm, write OSE data
    for iC = 1 : nC
        tic
        xiNorm2 = sum( xiOSE{ iC } .^ 2, 1 );
        tWall = toc;
        fprintf( logId, 'NORMXI realizationO %i/%i, batchO %i/%i, component %i/%i $2.4f \n', iR, nR, iB, nB, tWall );

        if Opt.ifWriteXi
            tic 
            setVelocity( obj( iC, iR ), xiNorm2, xiOSE{ iC }, iB )
            tWall = toc;
            fprintf( logId, 'WRITEXI realizationO %i/%i, batchO %i/%i, component %i/%i $2.4f \n', iR, nR, iB, nB, tWall );
        else
            tic 
            setVelocity( obj( iC, iR ), xiNorm2, [], iB )
            tWall = toc;
            fprintf( logId, 'WRITEXINORM realizationO %i/%i, batchO %i/%i, component %i/%i $2.4f \n', iR, nR, iB, nB, tWall );
        end
    end % component loop
end % out-of-sample global batch loop

clk = clock; % Exit gracefully
fprintf( logId, 'computeVelocity finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
