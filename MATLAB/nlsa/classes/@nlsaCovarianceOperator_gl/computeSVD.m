function [ u, s, v, aNorm  ] = computeSVD( obj, src, varargin )
% COMPUTESVD Singular value decomposition of an nlsaCovarianceOpeartor_gl
% object
% 
% Modified 2014/07/17


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup logfile and write calculation summary
Opt.logFile           = '';
Opt.logFilePermission = 'w';
Opt.logPath           = obj.path;
Opt.mode              = 'calcA';
Opt.method            = 'svdEcon';
Opt.ifWriteOperator   = true;
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
fprintf( logId, 'Mode %s \n', Opt.mode );
fprintf( logId, 'Method %s \n', Opt.method );
fprintf( logId, '----------------------------------------- \n' ); 

ifCV = false;
ifCU = false;

switch Opt.mode
    case 'calcA'
        if ~isempty( Opt.logFile )
            fclose( logId );
        end
        a = computeLinearMap( obj, src, ...
                              'logPath', Opt.logPath, ...  
                              'logFile', Opt.logFile, ...
                              'logFilePermission', 'a', ...
                              'ifWriteOperator', Opt.ifWriteOperator );
        if ~isempty( Opt.logFile )
            logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'a' );
        end
    case 'readA'
        tic
        a = getLinearMap( obj );
        tWall = toc;
        fprintf( logId, 'READA %^2.4f \n', tWall );
    case 'calcCT'
        if ~isempty( Opt.logFile )
            fclose( logId );
        end
        c = computeRightCovariance( obj, src, ...
                                    'logPath', Opt.logPath, ...  
                                    'logFile', Opt.logFile, ...
                                    'logFilePermission', 'a', ...
                                    'ifWriteOperator', Opt.ifWriteOperator );
        if ~isempty( Opt.logFile )
            logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'a' );
        end
        ifCV = true;
     case 'readCV'
        tic
        c = getRightCovariance( obj );
        tWall = toc;
        fprintf( logId, 'READCV %^2.4f \n', tWall );
        ifCV = true;
     case 'calcCU'
        if ~isempty( Opt.logFile )
            fclose( logId );
        end
        c = computeLeftCovariance( obj, src, ...
                                   'logPath', Opt.logPath, ...  
                                   'logFile', Opt.logFile, ...
                                   'logFilePermission', 'a', ...
                                   'ifWriteOperator', Opt.ifWriteOp );
        if ~isempty( Opt.logFile )
            logId = fopen( fullfile( Opt.logPath, Opt.logFile ), 'a' );
        end
        ifCU = true;
     case 'readCU'
        tic
        c = getLeftCovariance( obj );
        tWall = toc;
        fprintf( logId, 'READCU %^2.4f \n', tWall );
        ifCU = true;
 
end

tic
switch Opt.method
    case 'svdEcon'
        [ u, s, v ] = svd( a, 'econ' );
        s           = diag( s );
        tWall = toc;
        fprintf( logId, 'SVDECON %2.4f \n', tWall );  
    case 'svd'
        [ u, s, v ] = svd( a );
        s           = diag( s );
        fprintf( logId, 'SVD %2.4f \n', tWall );  
    case 'svds'
        [ u, s, v ] = svds( a, obj.nL );
        s           = diag( s );
        tWall = toc;
        fprintf( logId, 'SVDS %2.4f \n', tWall );  
    case 'eig'
        [ u, s ] = eig( c );
        s           = diag( s );
        [ s, idxSort ] = sort( s, 'descend' );
        u = u( :, idxSort );
        tWall = toc;
        fprintf( logId, 'EIG %2.4f \n', tWall );  
    case 'eigs'
        [ u, s ] = eigs( c, obj.nL );
        s           = diag( s );
        tWall = toc;
        fprintf( logId, 'EIGS %2.4f \n', tWall );  
end

if ifCU || ifCV
    s     = sqrt( s );
end

if ifCU
    tic
    v     = bsxfun( @rdivide, mltimes( src, u ), s' );
    vNorm = sqrt( sum( u .^ 2, 1 ) );
    v     = bsxfun( @rdivide, v, vNorm );
    tWall = toc;
    fprintf( logId, 'UV %2.4f \n', tWall );  
end
   
if ifCV
    tic
    v = u;
    u = bsxfun( @rdivide, mrtimes( src, v ), s' );
    uNorm = sqrt( sum( u .^ 2, 1 ) );
    u = bsxfun( @rdivide, u, uNorm );
    tWall = toc;
    fprintf( logId, 'UV %2.4f \n', tWall );  
end
 
nL = getNEigenfunction ( obj ); 
if nL < numel( s )
    s = s( 1 : nL );
    u = u( :, 1 : nL );
    v = v( :, 1 : nL );
end
aNorm  = sqrt( sum( s .^ 2 ) );
 
if Opt.ifWriteLeftSingularVectors
    setLeftSingularVectors( obj, u, '-v7.3' )
end
if Opt.ifWriteRightSingularVectors
    setRightSingularVectors( obj, v, '-v7.3' )
end
if Opt.ifWriteSingularValues
    setSingularValues( obj, s, '-v7.3' )
end

clk = clock; % Exit gracefully
fprintf( logId, 'computeSVD finished on %i/%i/%i %i:%i:%2.1f \n', ...
    clk( 1 ), clk( 2 ), clk( 3 ), clk( 4 ), clk( 5 ), clk( 6 ) );
if ~isempty( Opt.logFile )
    fclose( logId );
end
