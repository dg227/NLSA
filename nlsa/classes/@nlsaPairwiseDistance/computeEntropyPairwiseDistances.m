function [ yVal, yInd ] = computeEntropyPairwiseDistances( obj, diffOp, qry, iBatch, varargin )
% COMPUTEPAIRWISEDISTANCES_ENTROPY Compute pairwise distance from array of query 
% data and test data with entropy weights
% 
% Modified 2014/02/12


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments, set test data
if ~isa( diffOp, 'nlsaDiffusionOperator_ose' )
    error( 'Entropy of query data must be an nlsaDiffusionOperator_ose object' )
end
if isa( varargin{ 1 }, 'nlsaEmbeddedComponent' )
    if ~isa( varargin{ 2 }, 'nlsaDiffusionOperator_ose' )
        error( 'Entropy of test data must be specified as an nlsaDiffusionOperator_ose object' )
    end
    ifTst    = true;
    tst      = varargin{ 1 };
    diffOpT  = varargin{ 2 };
    varargin = varargin( 3 : end );
else
    ifTst          = false;
    tst            = qry;
    obj.partitionT = obj.partition;
    diffOpT        = diffOp;
end;
if ~isa( qry, 'nlsaEmbeddedComponent' )
    error( 'Query data must be specified as an array of nlsaEmbeddedComponent objects' )
end
if ~isa( tst, 'nlsaEmbeddedComponent' )
    error( 'Test data must be specified as an array of nlsaEmbeddedComponent objects' )
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate qry and tst partitions
ifC = isCompatible( qry, tst, 'testComponents', true ...
                              'testSamples', false );
if ~ifC
    error( 'Incompatible query and data partition' )
end
[ partitionQ, idxGQ ] = mergePartitions( getPartition( qry( 1, : ) ) );
[ partitionT, idxGT ] = mergePartitions( getPartition( tst( 1, : ) ) );
if ~isequal( mergePartitions( obj.partition ), partitionQ )
    error( 'Incompatible query partition' )
end
if ~isequal( mergePartitions( obj.partitionT ), partitionT )
    error( 'Incompatible test partition' )
end 

% idxGQ idxGT stores the global indices of the source partitions
[ nC, nRQ ] = size( qry );
[ ~,  nRT ] = size( tst );

nBQ = getNBatch( partitionQ ); % Number of batches from query data
nBT = getNBatch( partitionT ); % Number of batches from test data


% Get the maximum batch size for test data. It is used to determine the size
% of the distance matrix and nearest neighbor index matrix.
nSBMax = max( getBatchSize( partitionT ) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate partitions of the diffusion operator
if ~isequal( mergePartitions( getPartitionOSE( diffOp ) ), partitionQ )
    error( 'Incompatible diffusion operator partitions for query data' )
end
if ~isequal( mergePartitions( getPartitionOSE( diffOpT ) ), partitionT )
    error( 'Incompatible diffusion operator partitions for test data' )
end
s = getEntropyOSE( diffOp );
s = sqrt( 1 - exp( -2 * s ) );
if ifTst
    sT = getEntropyOSE( diffOpT );
    sT = sqrt( 1 - exp( -2 * sT ) );
else
    sT = s;
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nN = getNNeighbors( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments, setup logfile, and write calculation 
% summary
Opt.logFile = '';
Opt.logPath = obj.path;
Opt.ifWrite = true;
Opt.ifPartialSort = false;
Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'w' );
end
if Opt.ifPartialSort
    sortStr = 'MINK';
else
    sortStr = 'SORT';
end
if Opt.ifWrite
    pth = getDistancePath( obj );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computePairwiseDistances starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of query samples                 = %i, \n', getNSample( partitionQ ) );
fprintf( logId, 'Number of test samples                  = %i, \n', getNSample( partitionT ) );
fprintf( logId, 'Number of nearest neighbors             = %i, \n', nN );
fprintf( logId, 'Number of realizations for query        = %i, \n', nRQ );
fprintf( logId, 'Number of realizations for test         = %i, \n', nRT );
fprintf( logId, 'Number of query batches                 = %i, \n', getNBatch( partitionQ ) );
fprintf( logId, 'Number of test batches                  = %i, \n', nBT );   
fprintf( logId, 'Max batch size                          = %i, \n', nSBMax );
fprintf( logId, 'Min batch size                          = %i, \n', min( getBatchSize( partitionT ) ) );
fprintf( logId, 'Number of components                    = %i, \n', nC );
fprintf( logId, 'Max compoment physical space dimension  = %i, \n', max( getDimension( qry( :, 1 ) ) ) );
fprintf( logId, 'Min component physical space dimension  = %i, \n', min( getDimension( qry( :, 1 ) ) ) );
fprintf( logId, 'Max compoment embedding space dimension = %i, \n', max( getEmbeddingSpaceDimension( qry( :, 1 ) ) ) );
fprintf( logId, 'Min component embedding space dimension = %i, \n', min( getEmbeddingSpaceDimension( qry( :, 1 ) ) ) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the batches
% jG are the global indices of the query test batches


for iB = iBatch
    
    % Indices for query data
    iR  = idxGQ( 1, iB );
    iBR = idxGQ( 2, iB );
    nBR = getNBatch( qry( 1, iR ) );   
    nSI = getBatchSize( partitionQ, iB );
    iG  = getBatchLimit( partitionQ, iB );    

    fprintf( logId, '------------------------------------ \n' );
    fprintf( logId, 'Global query batch      %i/%i \n', iB, nBQ );
    fprintf( logId, 'Realization       %i/%i \n', iR, nRQ );
    fprintf( logId, 'Local batch       %i/%i \n', iBR, nBR );
    fprintf( logId, 'Number of samples %i \n', nSI );
    
    % Allocate distance arrays
    tic
    yVal = zeros( nSI, nN + nSBMax );          % Nearest neighbor distances % Need to change nSBMax
    yInd = zeros( nSI, nN + nSBMax, 'int32' ); % Nearest neighbor indices
    tWall = toc;
    fprintf( logId, 'ALLOCATE %i/%i %2.4f \n',  iB, nBQ, tWall );
    
    nN0 = 0;
    
    % Loop over the test batches
    for jB = 1 : nBT

        jR  = idxGT( 1, jB );
        jBR = idxGT( 2, jB );
        jG  = getBatchLimit( partitionT, jB ); 
        nSJ = getBatchSize( partitionT, jB  );
        yVal( :, nN0 + 1 : end ) = 0; 
                 
        % Loop over the components
        for iC = 1 : nC
       
            % Read query points 
            tic
            I = obj.getDistData( qry( iC, iR ), iBR );
            
            tWall = toc;
            fprintf( logId, 'READQ component %i/%i, realization %i/%i, local batch %i/%i, global batch %i/%i (%i samples) %2.4f \n', ...
                iC, nC, iR, nRQ, iBR, nBR, iB, nBQ, nSI, tWall );

            if ifTst || jB ~= iB
                % Read test points if needed
                tic
                J = obj.getDistData( tst( iC, jR ), jBR ); 
                tWall = toc;
                fprintf( logId, 'READT component %i/%i, realization %i/%i, local batch %i/%i, global batch %i/%i (%i samples) %2.4f \n', ...
                     iC, nC, jR, nRT, jBR, nBR, jB, nBT, nSJ, tWall );
            end
            
            % Compute entropy-weighted pairwise distance
            tic
            if ifTst || jB ~= iB
                yBatch = obj.evalDist( I, J );
                yBatch = bsxfun( @ldivide, s( iG( 1 ) : iG( 2 ) )', yBatch );
                yBatch = bsxfun( @rdivide, yBatch, sT( jG( 1 ) : jG( 2 ) ) );
            else
                yBatch = obj.evalDist( I );
                yBatch = bsxfun( @ldivide, s( iG( 1 ) : iG( 2 ) )', yBatch );
                yBatch = bsxfun( @rdivide, yBatch, s( iG( 1 ) : iG( 2 ) ) );
            end
            tWall = toc;
            fprintf( logId, 'DMAT component %i/%i, realization %i-%i, local batches %i-%i, global batches %i-%i (embedding space dimension %i) %2.4f \n', ...
                iC, nC, iR, jR, iBR, jBR, iB, jB, tWall );
            yVal( :, nN0 + 1 : nN0 + nSJ ) = yVal( :, nN0 + 1 : nN0 + nSJ ) + yBatch;
        end

        yInd( :, nN0 + 1 : nN0 + nSJ ) = int32( repmat( jG( 1 ) : jG( 2 ), [ nSI 1 ] ) );
        
             
        % Sort nearest neighbors
        tic
        nSort = min( obj.nN, nN0 + nSJ );
        if Opt.ifPartialSort
            [ yVal( :, 1 : nSort ), idx ] = mink( yVal( :, 1 : nN0 + nSJ ), nSort, 2 ); % operate along columns
        else
            [ tmp, idx ] = sort( yVal( :, 1 : nN0 + nSJ ), 2, 'ascend' );
            yVal( :, 1 : nSort ) = tmp( :, 1 : nSort );
            idx                  = idx( :, 1 : nSort );
        end
        for iSI = 1 : nSI
            yInd( iSI, 1 : nSort )  = yInd( iSI, idx( iSI, : ) );
        end
        tWall = toc;
        fprintf( logId, '%s component %i/%i, realization %i-%i, local batches %i-%i, global batches %i-%i (%i sorted samples)  %2.4f \n', ...
             sortStr, iC, nC, iR, iR, iBR, iBR, iB, iB, nSort, tWall );
         
        nN0 = nSort;
    end
    
    yVal = yVal( :, 1 : nN );
    yInd = yInd( :, 1 : nN );
    if ~ifTst
        yVal( :, 1 ) = 0;                        % iron out numerical wrinkles 
    end
    if Opt.ifWrite
        save( fullfile( pth, obj.file{ iR }{ iB } ), '-v7.3', 'yVal', 'yInd' )
    end
end

clk = clock; % Exit gracefully
fprintf( logId, 'computePairwiseDistances_entropy finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end    
