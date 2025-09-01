function computeVelocityError( obj, src, varargin )
% COMPUTEVELOCITYERROR Compute out-of-sample extension error of the phase 
% space velocity of an array of nlsaEmbeddedComponent_ose objects relative 
% to a reference array of nlsaEmbeddedComponent_xi objects
%
% Modified 2014/04/24

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate input arguments
msgId = [ obj.getErrMsgId ':computeVelocityError:' ];
if ~isa( src, 'nlsaEmbeddedComponent_xi' )
    msgStr = 'Second argument must be an nlsaEmbeddedComponent_xi object.';
    error( [ msgId 'invalidSrc' ], msgStr )
end
if ~isCompatible( obj, src )
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
Opt.logPath   = getVelocityErrorPath( obj );
Opt.ifWriteXi = false;
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
fprintf( logId, 'computeVelocityError starting on %i/%i/%i %i:%i:%2.1f \n', ...
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
        % Read source and OSE phase space velocity 
        tic
        [ xiNorm2, xi ] = getVelocity( src( iC, iR ), iB );
        tWall = toc;
        fprintf( logId, 'READXI realization %i/%i, batch %i/%i, component %i/%i, %2.4f', iR, nR, iB, nB, iC, nC, tWall );
        tic
        [ ~, xiE ] = getVelocity( obj( iC, iR ), iB );
        tWall = toc;
        fprintf( logId, 'READXIOSE realization %i/%i, batch %i/%i, component %i/%i, %2.4f', iR, nR, iB, nB, iC, nC, tWall );
            
        % Compute error
        tic
        xiE      = xiE - xi;
        xiENorm2 = sum( xiE .^ 2, 1 );
        tWall = toc;
        fprintf( logId, 'ERRORXI realization %i/%i, batch %i/%i, component %i/%i, %2.4f', iR, nR, iB, nB, iC, nC, tWall );

        % Write out results
        if Opt.ifWriteXi
            tic
            setVelocityError( obj( iC, iR ), xiENorm2, xiE, xiNorm2, iB )
            tWall = toc;
            fprintf( logId, 'WRITEERRORXI realizationO %i/%i, batchO %i/%i, component %i/%i $2.4f \n', iR, nR, iB, nB, iC, nC, tWall );
        else
            tic
            setVelocityError( obj( iC, iR ), xiENorm2, [], xiNorm2, iB )
            tWall = toc;
            fprintf( logId, 'WRITEERRORXINORM realizationO %i/%i, batchO %i/%i, component %i/%i $2.4f \n', iR, nR, iB, nB, iC, nC, tWall );
        end
    end % component loop
end % global batch loop

clk = clock; % Exit gracefully
fprintf( logId, 'computeVelocityError finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
