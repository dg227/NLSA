function [ yVal, yInd ] = computePairwiseDistances( obj, qry, varargin )
% COMPUTEPAIRWISEDISTANCES Compute pairwise distance from array of query 
% data and test data
% 
% Modified 2019/10/20

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments, set test data
dFunc = getLocalDistanceFunction( obj );

%if ~iscompatible( lDist, qry )
%    error( 'Query data are inompatible with the local distance function' ) 
%end
%if ~iscompatible( qry )
%    error( 'Incompatible query data' )
%end

if ~ischar( varargin{ 1 } )
    ifTst    = true;
    tst      = varargin{ 1 };
    varargin = varargin( 2 : end );
 
%    if ~iscompatible( lDist, tst )
%        error( 'Test data are inompatible with the local distance function' ) 
%    end
%    if ~isCompatible( tst )
%        error( 'Incompatible test data' )
%    end
%    if ~isCompatible( qry, tst, 'testComponents', true, ...
%                                'testSamples',    false );
%        error( 'Incompatible query and test data' )
%    end
else
    ifTst  = false;
    tst    = qry;
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup and validate query and test partitions
partitionQ = getPartition( qry );
partitionT = getPartition( tst );
nBR        = getNBatch( partitionQ );
nBRT       = getNBatch( partitionT );
[ partitionQ, idxGQ ] = mergePartitions( partitionQ );
[ partitionT, idxGT ] = mergePartitions( partitionT );
if ~isequal( mergePartitions( getPartition( obj ) ), partitionQ )
    error( 'Incompatible query partition' )
end
if ~isequal( mergePartitions( getPartitionTest( obj ) ), partitionT )
    error( 'Incompatible test partition' )
end 
nC  = getNComponent( qry );
nRQ = getNRealization( qry );
nRT = getNRealization( tst );
nBQ = getNBatch( partitionQ ); % Number of batches from query data
nBT = getNBatch( partitionT ); % Number of batches from test data

% Get the maximum batch size for test data. It is used to determine the size
% of the distance matrix and nearest neighbor index matrix.
nSBMax = max( getBatchSize( partitionT ) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nearest neighbors to be retained
nN = getNNeighbors( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments, setup logfile, and write calculation 
% summary
Opt.batch   = 1 : getNBatch( partitionQ );
Opt.logFile = '';
Opt.logPath = obj.path;
Opt.ifWrite = true;
Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'w' );
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
fprintf( logId, 'Local distance function class           = %s  \n', class( dFunc ) );
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
fprintf( logId, 'Max compoment physical space dimension  = %i, \n', max( getPhysicalSpaceDimension( qry ) ) );
fprintf( logId, 'Min component physical space dimension  = %i, \n', min( getPhysicalSpaceDimension( qry ) ) );
fprintf( logId, 'Max compoment embedding space dimension = %i, \n', max( getEmbeddingSpaceDimension( qry ) ) );
fprintf( logId, 'Min component embedding space dimension = %i, \n', min( getEmbeddingSpaceDimension( qry ) ) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the batches
% jG are the global indices of the query test batches


for iB = Opt.batch
    
    % Indices for query data
    iR  = idxGQ( 1, iB );
    iBR = idxGQ( 2, iB );
    nSI = getBatchSize( partitionQ, iB );
    
    fprintf( logId, '------------------------------------ \n' );
    fprintf( logId, 'Global query batch      %i/%i \n', iB, nBQ );
    fprintf( logId, 'Realization       %i/%i \n', iR, nRQ );
    fprintf( logId, 'Local batch       %i/%i \n', iBR, nBR );
    fprintf( logId, 'Number of samples %i \n', nSI );
    
    % Allocate distance arrays
    tWall = tic;
    yVal = zeros( nSI, nN + nSBMax );          % Nearest neighbor distances % Need to change nSBMax
    yInd = zeros( nSI, nN + nSBMax, 'int32' ); % Nearest neighbor indices
    tWall = toc( tWall );
    fprintf( logId, 'ALLOCATE %i/%i %2.4f \n',  iB, nBQ, tWall );
    
    nN0 = 0;
    
    % Loop over the test batches
    for jB = 1 : nBT

        jR   = idxGT( 1, jB );
        jBR  = idxGT( 2, jB );
        jG   = getBatchLimit( partitionT, jB ); 
        nSJ  = getBatchSize( partitionT, jB  );
        yVal( :, nN0 + 1 : end ) = 0; 
                 
        % Loop over the components
        for iC = 1 : nC
       
            % Read query points 
            if nC > 1 || jB == 1 
                tWall = tic;
                dFunc = importQueryData( dFunc, qry, iC, iBR, iR );
                tWall = toc( tWall );
                fprintf( logId, 'READQ component %i/%i, realization %i/%i, local batch %i/%i, global batch %i/%i (%i samples) %2.4f \n', ...
                    iC, nC, iR, nRQ, iBR, nBR( iR ), iB, nBQ, nSI, tWall );
            end
            
            if ifTst || jB ~= iB
                % Read test points if needed
                tWall = tic; 
                dFunc = importTestData( dFunc, tst, iC, jBR, jR ); 
                tWall = toc( tWall );
                fprintf( logId, 'READT component %i/%i, realization %i/%i, local batch %i/%i, global batch %i/%i (%i samples) %2.4f \n', ...
                     iC, nC, jR, nRT, jBR, nBRT( jR ), jB, nBT, nSJ, tWall );
            end
            
            % Compute pairwise distance
            tWall = tic;
            if ifTst || jB ~= iB
                yBatch = evaluateDistance( dFunc );
            else
                yBatch = evaluateDistance( dFunc, 'self' );
            end
            tWall = toc( tWall );
            fprintf( logId, 'DMAT component %i/%i, realization %i-%i, local batches %i-%i, global batches %i-%i %2.4f \n', ...
                iC, nC, iR, jR, iBR, jBR, iB, jB, tWall );
            yVal( :, nN0 + 1 : nN0 + nSJ ) = yVal( :, nN0 + 1 : nN0 + nSJ ) + yBatch;
        end

        yInd( :, nN0 + 1 : nN0 + nSJ ) = int32( repmat( jG( 1 ) : jG( 2 ), [ nSI 1 ] ) );
        
        % Sort nearest neighbors
        tWall = tic;
        nSort = min( obj.nN, nN0 + nSJ );
        [ yVal( :, 1 : nSort ), idx ] = mink( yVal( :, 1 : nN0 + nSJ ), nSort, 2 ); % operate along columns
        kdx = sub2ind( [ nSI ( nN0 + nSJ ) ], repmat( ( 1 : nSI )', [ 1 nSort ] ), idx );  
        yInd( :, 1 : nSort )  = yInd( kdx );
        tWall = toc( tWall );
        fprintf( logId, 'MINK component %i/%i, realization %i-%i, local batches %i-%i, global batches %i-%i (%i sorted samples)  %2.4f \n', ...
             iC, nC, iR, jR, iBR, jBR, iB, jB, nSort, tWall );
        nN0 = nSort;
    end
    
    yVal = yVal( :, 1 : nN );
    yInd = yInd( :, 1 : nN );
    if ~ifTst
        yVal( :, 1 ) = 0;                        % iron out numerical wrinkles 
    end
    if Opt.ifWrite
        tWall = tic;
        setDistances( obj, yVal, yInd, iBR, iR, '-v7.3' )
        tWall = toc( tWall );
        fprintf( logId, 'WRITEDIST realization %i/%i, local batch %i/%i, global batch %i/%i  %2.4f \n', ...
            iR, nRQ, iBR, nBR, iB, nBQ, tWall ); 
    end
end

clk = clock; % Exit gracefully
fprintf( logId, 'computePairwiseDistances finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end


    
