function computeData( obj, src1, src2, varargin )
% COMPUTEDATA Compute data difference from nlsaEmbeddedComponent objects
% src1 and src2
%
% Modified 2014/07/10

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
msgId = [ obj.getErrMsgId ':computeData:' ];
if ~isa( src1, 'nlsaEmbeddedComponent' )
    msgStr = 'Second argument must be an nlsaEmbeddedComponent object.';
    error( [ msgId 'invalidSrc' ], msgStr )
end
if ~isa( src2, 'nlsaEmbeddedComponent' )
    msgStr = 'Third argument must be an nlsaEmbeddedComponent object.';
end
if ~isCompatible( obj, src1 )
    msgStr = 'Incompatible OSE and source components';
    error( [ msgId 'incompatibleComp' ], msgStr ) 
end
if ~isCompatible( obj, src2 )
    msgStr = 'Incompatible OSE and source components';
    error( [ msgId 'incompatibleComp' ], msgStr ) 
end
[ nC, nR ] = size( obj );
partition = getPartition( obj( 1, : ) );
[ partitionG,  idxBG  ] = mergePartitions( partition );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.batch     = 1 : getNBatch( partitionG );
Opt.logFile   = '';
Opt.logPath   = getStateErrorPath( obj );
Opt.ifWriteX  = false;
Opt = parseargs( Opt, varargin{ : } );

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
fprintf( logId, 'Number of components    = %i, \n', nC );
fprintf( logId, 'Number of realizations  = %i, \n', nR );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the realizations and components
for iBG = Opt.batch

    iR  = idxBG( 1, iBG );
    iB  = idxBG( 2, iBG );
    nB  = getNBatch( partition( iR ) );
    nSB = getBatchSize( partition( iR ), iB );

    for iC = 1 : nC
        % Read source data
        tic
        xE = getData( src1( iC, iR ), iB );
        tWall = toc;
        fprintf( logId, 'READ1 realization %i/%i, batch %i/%i, component %i/%i, %2.4f', iR, nR, iB, nB, iC, nC, tWall );
        tic
        x = getData( src2( iC, iR ), iB );
        tWall = toc;
        fprintf( logId, 'READ2 realization %i/%i, batch %i/%i, component %i/%i, %2.4f', iR, nR, iB, nB, iC, nC, tWall );
           
        % Compute error
        tic
        xE      = xE - x;
        xENorm2 = sum( xE .^ 2, 1 );
        tWall = toc;
        fprintf( logId, 'DIFF realization %i/%i, batch %i/%i, component %i/%i, %2.4f', iR, nR, iB, nB, iC, nC, tWall );

        % Write out results
        if Opt.ifWriteX
            tic
            setData( obj( iC, iR ), xENorm2, xE, iB )
            tWall = toc;
            fprintf( logId, 'WRITED realization %i/%i, batch %i/%i, component %i/%i $2.4f \n', iR, nR, iB, nB, iC, nC, tWall );
        else
            tic
            setData( obj( iC, iR ), xENorm2, [], iB )
            tWall = toc;
            fprintf( logId, 'WRITEDNORM realization %i/%i, batch %i/%i, component %i/%i $2.4f \n', iR, nR, iB, nB, iC, nC, tWall );
        end
    end % component loop
end % global batch loop

clk = clock; % Exit gracefully
fprintf( logId, 'computeData finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
