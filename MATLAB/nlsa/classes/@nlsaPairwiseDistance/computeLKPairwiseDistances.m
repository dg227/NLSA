function [ yVal, yInd ] = computeLKPairwiseDistances( obj, sDist, diffOp, iBatch, varargin )
% COMPUTELKPAIRWISEDISTANCES Compute Lukaszyk-Karmowski pairwise distances from 
% symmetric distances and transition probabilities
% 
% Modified 2014/02/14


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments, set test data
if ~isa( pDist, 'nlsaSymmetricDistance' )
    error( 'Distance data must be specified as an nlsaSymmetric distance object' )
end
if ~isa( diffOp, 'nlsaDiffusionOperator_ose' )
    error( 'Transition probabilities must specified via an nlsaDiffusionOperator_ose object' )
end
[ partition, idxG ] = mergePartitions( getPartition( obj ) );
if ~isequal( partition, mergePartitions( getPartitionOSE( diffOp ) );
    error( 'Incompatible out-of-sample partitions )
end
nS   = sum( getNSample( partition ) );
nSIn = getNSample( sDist );

if nSIn ~= getNSample( diffOp ) 
    error( 'Incompatible number of in-sample data' )
end

nR     = numel( getPartition( obj ) );
nB     = getNBatch( partition ); % Number of out-of-sample batches
nSBMax = max( getBatchSize( partition ) );
nN     = getNNeighbors( obj );

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
fprintf( logId, 'computeLKPairwiseDistances starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of in- samples                 = %i, \n', nSIn );
fprintf( logId, 'Number of out- samples                = %i, \n', nS );
fprintf( logId, 'Number of nearest neighbors           = %i, \n', nN );
fprintf( logId, 'Number of realizations for OSE data   = %i, \n', nR );
fprintf( logId, 'Number of batches for OSE data        = %i, \n', nB );
fprintf( logId, 'Max batch size                          = %i, \n', nSBMax );
fprintf( logId, 'Min batch size                          = %i, \n', min( getBatchSize( partition ) ) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read symmetrized distance data and form kernel for the in-sample data
tic
[ yVal, yRow, yCol ] = getData( src );
yRow = double( yRow );
yCol = double( yCol );
nNZ = numel( yVal );
tWall = toc;
fprintf( logId, 'READS %i samples, %i matrix elements %2.4f \n', ...
         nSIn, nNZ, tWall );
tic
yVal = exp( -yVal / obj.epsilon ^ 2 ); % yVal is distance ^ 2
tWall = toc;
fprintf( logId, 'EXPS, %i nonzero entries, %2.4f avg. per row %2.4f, \n', ...
         nNZ, nNZ / nSIn, tWall );



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the OSE batches
for iB = iBatch
    
    iR  = idxG( 1, iB );
    iBR = idxG( 2, iB );
    nSI = getBatchSize( partition, iB );
    iG  = getBatchLimit( partition, iB );    

    fprintf( logId, '------------------------------------ \n' );
    fprintf( logId, 'Global OSE batch  %i/%i \n', iB, nB  );
    fprintf( logId, 'Realization       %i/%i \n', iR, nR  );
    fprintf( logId, 'Local batch       %i \n', iBR  );
    fprintf( logId, 'Number of samples %i \n', nSI );
    
    % Allocate distance arrays
    tic
    yVal = zeros( nSI, nN + nSBMax );          % Nearest neighbor distances % Need to change nSBMax
    yInd = zeros( nSI, nN + nSBMax, 'int32' ); % Nearest neighbor indices
    tWall = toc;
    fprintf( logId, 'ALLOCATE %i/%i %2.4f \n',  iB, nBO, tWall );
    
    nN0 = 0;
    
    % Read transition probabilities
    tic
    [ pI, subI ] = getOperatorOSE( obj, iBR, iR );
    tWall = toc
    fprintf( logId, 'READP %i %i %2.4f \n', iBR, iR, tWall );

    % Loop over the OSE batches
    for jB = 1 : nBO

        jR  = idxGO( 1, jB );
        jBR = idxGO( 2, jB );
        jG  = getBatchLimit( partition, jB ); 
        nSJ = getBatchSize( partition, jB  );
        yVal( :, nN0 + 1 : end ) = 0; 
        
        % Read transition probabilities
        tic
        [ pJ, subJ ] = getOperatorOSE( obj, jBR, jR );
        tWall = toc;
        fprintf( logId, 'READP %i %i %2.4f \n', jBR, jR, tWall );

        
        % Loop over samples 
        for iS = 1 : nSI
            for jS = 1 : nSJ
                subIJ = [ subI( iS, : )' subJ( jS, : )' ];
                [ ~, indIJ ] = intersect( yRowCol, subIJ );
                yBatch( iS,jS ) = sum( yVal( indIJ ) );
            end
        end

        
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
