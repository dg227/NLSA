function [ u, s, v, aNorm  ] = computeSVD( obj, prj, varargin )
% COMPUTESVD Singular value decomposition of nlsaLinearMap objects
% 
% Modified 2015/10/19


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check input arguments
nA = numel( obj );
if nA > 1 && nargout >= 1
    error( 'Explicit output is not supported for non-scalar nlsaLinearMap objects' )
end
if ~isCompatible( obj )
    error( 'Incompatible linear maps' )
end
if nargin > 2 && ~isempty( prj )
    if ~isCompatible( obj, prj )
        error( 'Incompatible projected components.' )
    end
end
nC = getNComponent( obj( 1 ) );
nD = getDimension( obj( 1 ) );
nV = getNEigenfunction( obj );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
Opt.logFile           = '';
Opt.logFilePermission = 'w';
Opt.logPath           = getOperatorPath( obj( end )  );;
Opt.mode              = 'calcA';
Opt.ifWriteOperator             = true;
Opt.ifWriteLeftSingularVectors  = true;
Opt.ifWriteRightSingularVectors = true;
Opt.ifWriteSingularValues       = true;
Opt = parseargs( Opt, varargin{ : } );

if isempty( Opt.logFile )
    logId = 1;
else
    logId = fopen( fullfile( Opt.logPath, Opt.logFile ), Opt.logFilePermission );
end

clk = clock;
[ ~, hostname ] = unix( 'hostname' );
fprintf( logId, 'computeSVD starting on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
fprintf( logId, 'Hostname %s \n', hostname );
fprintf( logId, 'Path %s \n', obj.path );
fprintf( logId, '----------------------------------------- \n' ); 

switch Opt.mode
    case 'calcA'
        if ~isempty( Opt.logFile )
            fclose( logId );
        end
        a = computeLinearMap( obj( end ), prj, ...
                              'logPath', Opt.logPath, ...  
                              'logFile', Opt.logFile, ...
                              'logFilePermission', 'a', ...
                              'ifWriteOperator', Opt.ifWriteOperator );
        if ~isempty( Opt.logFile ) 
            logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'a' );
        end
    case 'readA'
        tic
        a = getLinearMap( obj( nA ) );
        tWall = toc;
        fprintf( logId, 'READA %^2.4f \n', tWall );
end

nA = numel( obj );
for iA = nA : -1 : 1
    if iA < nA
        tic 
        a = a( :, 1 : iA );
        tWall = toc;
        fprintf( logId, 'TRIMA %i -> %i %2.4f \n', ...
            getNEigenfunction( obj( iA + 1 ) ), ...
            getNEigenfunction( obj( iA ) ), tWall );  
    end
 
    tic
    [ u, s, v ] = svd( a, 0 );
    s           = diag( s );
    aNorm       = sqrt( sum( s .^ 2 ) );
    tWall = toc;
    fprintf( logId, 'SVD %2.4f \n', tWall );  
 
    pth = fullfile( obj( iA ).path, obj( iA ).pathA );
    if Opt.ifWriteLeftSingularVectors
        setLeftSingularVectors( obj( iA ), u, 1 : nC, '-v7.3' )
    end
    if Opt.ifWriteRightSingularVectors
        setRightSingularVectors( obj( iA ), v, '-v7.3' )
    end
    if Opt.ifWriteSingularValues
        setSingularValues( obj( iA ), s )
    end
end 

clk = clock; % Exit gracefully
fprintf( logId, 'computeSVD finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
