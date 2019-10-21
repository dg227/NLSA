function symmetrizeDistances( obj, pDist, varargin )
% Compute symmetric distance matrix from pairwise distances 
% 
% Modified 2014/06/13


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
if ~isa( pDist, 'nlsaPairwiseDistance' )
    error( 'Second argument must be an nlsaPairwiseDistance object' )
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup and validate partitions
partition = getPartition( obj );
if any( ~isequal( partition, getPartition( pDist ) ) )
    error( 'Incompatible pairwise distance partition' )
end
if any( ~isequal( partition, getPartitionTest( pDist ) ) )
    error( 'The query and test partitions of the pairwise distance data must be equal' )
end
[ partitionG, idxG ] = mergePartitions( partition );
nR = numel( partition );
nB = getNBatch( partitionG );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nearest neighbors to be retained
nN    = getNNeighbors( obj );
if nN > getNNeighbors( pDist )
    error( 'Number of symmetric nearest neighbors cannot exceed the number of pairwise nearest neighbors' )
end
nNMax = getNNeighborsMax( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments, setup logfile, and write calculation 
% summary
Opt.batch   = 1 : nB;
Opt.logFile = '';
Opt.logPath = obj.path;
Opt.ifSort  = false;
Opt = parseargs( Opt, varargin{ : } );
if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'w' );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'symmetrizeDistances starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, 'Number of realizations              = %i, \n', nR );
fprintf( logId, 'Number of batches                   = %i, \n', getNBatch( partitionG ) );
fprintf( logId, 'Number of samples                   = %i, \n', getNSample( partitionG ) );
fprintf( logId, 'Max batch size                      = %i, \n', max( getBatchSize( partitionG ) ) );
fprintf( logId, 'Min batch size                      = %i, \n', min( getBatchSize( partitionG ) ) );
fprintf( logId, 'Number of nearest neighbors         = %i, \n', nN );
fprintf( logId, 'Maximum number of nearest neighbors = %i, \n', nN );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the batches
for iB = Opt.batch
    
    % Realization-local indices 
    iR  = idxG( 1, iB );
    iBR = idxG( 2, iB );
    nBI = getNBatch( partition( iR ) );   
    nSI = getBatchSize( partition( iR ), iBR );
    iG  = getBatchLimit( partitionG, iB ); 

    fprintf( logId, '------------------------------------ \n' );
    fprintf( logId, 'Global batch      %i/%i \n', iB, nB );
    fprintf( logId, 'Realization       %i/%i \n', iR, nR );
    fprintf( logId, 'Local batch       %i/%i \n', iBR, nBI );
    fprintf( logId, 'Number of samples %i \n', nSI );
    
    % Allocate distance arrays
    tic
    yValI = ones( nSI, nNMax ) * Inf;    % Distances
    yIndI = ones( nSI, nNMax, 'int32' ); % Nearest neighbor indices
    tWall = toc;
    fprintf( logId, 'ALLOCATE %i/%i %2.4f \n',  iB, nB, tWall );
    
    % Read pairwise distance data
    tic
    [ yValJ, yIndJ ] = getDistances( pDist, iBR, iR ); 
    yValI( :, 1 : nN ) = yValJ( :, 1 : nN );
    yIndI( :, 1 : nN ) = yIndJ( :, 1 : nN );
    iNN = ones( nSI, 1 ) * ( nN + 1 );
    tWall = toc;
    fprintf( logId, 'READI realization %i/%i, local batch %i/%i, global batch %i/%i %2.4f \n', ...
        iR, nR, iBR, nBI, iB, nB, tWall );
    
    % Loop over the pairwise distance batches
    for jB = 1 : nB

        jR   = idxG( 1, jB );
        jBR  = idxG( 2, jB );
        nBJ  = getNBatch( partition( jR ) );
        nSJ  = getBatchSize( partition( jR ), jBR );
        jG   = getBatchLimit( partitionG, jB ); 
                 
        % Read pairwise distance data
        tic
        [ yValJ, yIndJ ] = getDistances( pDist, jBR, jR );
        yValJ = yValJ( :, 1 : nN );
        yIndJ = yIndJ( :, 1 : nN );
        tWall = toc;
        fprintf( logId, 'READJ realization %i/%i, local batch %i/%i, global batch %i/%i %2.4f \n', ...
            jR, nR, jBR, nBJ, jB, nB, tWall );

        % Identify samples from batch J to be added to the neighborhood of
        % batch I
        %
        % ifJI:   logical array selecting the samples in batch J which have
        %         nearest neighbors in batch I
        % rowI:   lists the samples (rows) in batch I selected by ifJI
        % rowJ:   lists the samples (rows) in batch J selected by ifJI
        % nnJ:    lists the nearest neighbor indices in batch J corresponding
        %         to the samples selected by ifJI
        % ifAdd:  logical array trimming the samples in batch J selected by ifJI
        %         to select those which are not in the nearest neighborhoods 
        %         yIndIJ

        tic
        nAddTot = 0;
        ifJI          = yIndJ >= iG( 1 ) & yIndJ <= iG( 2 );
        rowI          = yIndJ( ifJI ) - iG( 1 ) + 1;
        [ rowJ, nnJ ] = find( ifJI );

        % indAdd identifies the samples selected by ifJI such that J is not
        % in the neighborhood of I
        % rowJ + jG( 1 ) - 1:    global indices of the selected samples in J
        % yIndI( rowI, 1 : nN ): global indices of the selected samples in I
        indAdd = find( ~any( bsxfun( @eq, rowJ + jG( 1 ) - 1, ...
                                     yIndI( rowI, 1 : nN ) ), 2 ) );              
        nAdd   = numel( indAdd );
        for iAdd = 1 : nAdd
            iI = rowI( indAdd( iAdd ) );
            jI = iNN( iI );
            iJ = rowJ( indAdd( iAdd ) );
            jJ = nnJ( indAdd( iAdd ) );

            yValI( iI, jI ) = yValJ( iJ, jJ );
            yIndI( iI, jI ) = iJ + jG( 1 ) - 1;
            iNN( iI )       = iNN( iI ) + 1;

            if iNN( iI ) > nNMax
                error( 'Maximum number of nearest neighbors exceeded' )
            end
        end
        nAddTot = nAddTot + nAdd; 
        tWall = toc;
        fprintf( logId, 'SYM %i %2.4f \n', nAddTot, tWall );
    end % test batch loop

    % Sort distances
    if Opt.ifSort
        tic    
        [ yValI, idxSort ] = sort( yValI, 2, 'ascend' );
        for iS = 1 : nSI
            yIndI( iS, : ) = yIndI( iS, idxSort( iS, : ) );
        end
        tWall = toc;
        fprintf( logId, 'SORT realization %i-%i, local batches %i-%i, global batches %i-%i %2.4f \n', ...
                 iR, nR, iBR, iBR, iB, iB, tWall );
    end

    % Write data
    tic 
    setDistances( obj, yValI, yIndI, iBR, iR, '-v7.3' )
    tWall = toc;
    fprintf( logId, 'WRITE realization %i-%i, local batches %i-%i, global batches %i-%i %2.4f \n', ...
             iR, nR, iBR, iBR, iB, iB, tWall );

end % query batch loop
        

clk = clock; % Exit gracefully
fprintf( logId, 'symmetrizeDistances finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end


    
