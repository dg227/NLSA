function [ model, In, Out ] = l63NLSAModel( experiment )
% L63NLSAModel Build NLSA model for Lorenz 63 system
%
%  [ model, In, Out ] = l63NLSAModel( experiment ) builds an NLSA model based 
%  on the string identifier experiment.  
%
%  The model parameters for the in-sample (training) and, optionally, 
%  out-of-sample (test) data are specified in the structures In and Out. 
%
%  l63NLSAModel converts the parameter values in In and Out into appropriate 
%  arguments for the nlsaModel class constructors, and then calls the 
%  constructors to build the model.
%
%  For additional information on the class constructors see 
%
%      /nlsa/classes/nlsaModel_base/nlsaModel_base.m
%      /nlsa/classes/nlsaModel/nlsaModel.m
%      /nlsa/classes/nlsaModel_den/nlsaModel_den.m
%      /nlsa/classes/nlsaModel_den_ose/nlsaModel_den_ose.m
%
% Modified 2020/03/30
 
if nargin == 0
    experiment = '6.4k';
end

switch experiment

    % 6400 samples, standard L63 parameters
    case '6.4k'
        % In-sample dataset parameters
        In.dt         = 0.01;  % sampling interval
        In.Res.beta   = 8/3;   % L63 parameter beta
        In.Res.rho    = 28;    % L63 parameter rho
        In.Res.sigma  = 10;    % L63 parameter sigma
        In.Res.nSProd = 6400;  % number of "production samples
        In.Res.nSSpin = 64000; % spinup samples
        In.Res.x0     = [ 0 1 1.05 ]; % initial conditions
        In.Res.relTol = 1E-8;  % relative tolerance for ODE solver 
        In.Res.ifCent = false; % data centering

        % Source data
        In.Src.idxX    = 1 : 3;       % observed state vector components 
        In.Src.idxE    = 1 : 1;       % delay embedding indices
        In.Src.nXB     = 1;           % additional samples before main interval
        In.Src.nXA     = 500;         % additional samples after main interval
        In.Src.fdOrder = 2;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'overlap'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 3;     % observed state vector components 
        In.Trg.nEL       = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 1;         % additional samples before main interval
        In.Trg.nXA       = 500;       % additional samples after main interval
        In.Trg.fdOrder   = 2;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'overlap'; % storage format for delay embedding

        % Out-of-sample dataset parameters
        Out.Res.beta   = 8/3;   % L63 parameter beta
        Out.Res.rho    = 28;    % L63 parameter rho
        Out.Res.sigma  = 10;    % L63 parameter sigma
        Out.Res.nSprod = 6400;  % number of "production samples
        Out.Res.nSSpin = 128000; % spinup samples
        Out.Res.relTol = 1E-8;  % relative tolerance for ODE solver 
        Out.Res.ifCent = false; % data centering
        Out. nS     =   Out.Res.nSprod + Out.nEL + Out.nXB + Out.nXA; % sample number

        % NLSA parameters
        In.nB         = 1;          % batches to partition the in-sample data
        In.nBRec      = In.nB;      % batches for reconstructed data
        In.nN         = 5000;       % nearest neighbors for pairwise distances
        In.lDist      = 'l2';       % local distance
        In.tol        = 0;          % 0 distance threshold (for cone kernel)
        In.zeta       = 0;          % cone kernel parameter 
        In.coneAlpha  = 0;          % velocity exponent in cone kernel
        In.nNS        = In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb_bs'; % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -20 20 ];% kernel bandwidth exponents 
        In.nEpsilon    = 200;       % number of exponents for bandwidth tuning
        In.alpha       = .5;        % diffusion maps normalization 
        In.nPhi        = 2001;      % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType     = 'vb';          % density estimation type
        In.denND       = 2;             % manifold dimension for 
        In.denLDist    = 'l2';          % local distance function 
        In.denBeta     = -1 / In.denND; % density exponent 
        In.denNN       = 8;             % nearest neighbors 
        In.denZeta     = 0;             % cone kernel parameter 
        In.denAlpha    = 0;             % cone kernel velocity exponent 
        In.denEpsilonB = 2;             % kernel bandwidth base 
        In.denEpsilonE = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon = 200;        % number of exponents for bandwidth tuning

        % NLSA parameters, out-of-sample data
        Out.nB    = 1;         % bathches to partition the in-sample data
        Out.nBRec = Out.nB;    % batches for reconstructed data

    otherwise
        error( 'Invalid experiment' )
end


%% PRELIMINARY CHECKS
% Check if we are doing out-of-sample extension
if ~exist( Out, 'var' )
    Out   = [];
    ifOse = false; 
else
    ifOse = true;
end

% Check if we are using kernel density estimation
if isfield( In, 'denType' )
    ifDen = true;
else
    ifDen = false;
end

% Check that required high-level fields are present
if ~isfield( In, 'Src' )
    error( 'Source field Src missing from in-sample parameters.' )
end
if ~isfield( In, 'Res' )
    error( 'Realization field Res missing from in-sample parameters.' )
end
if ifOse && ~isfield( Out, 'Res' )
    warning( [ 'Realization field Res missing from  out-of-sample ' ...
               'parameters. Reverting to default from in-sample data.' ] )
end


%% ROOT DIRECTORY NAMES
% In-sample data
if isfield( In, 'dataPath' )
    inPath = In.dataPath;
else
    inPath      = fullfile( pwd, 'data/raw' ); 
    In.dataPath = inPath;
end

% Out-of-sample data
if isfield( Out, 'dataPath' )
    outPath = Out.dataPath;
else
    outPath      = fullfile( pwd, 'data/raw' );
    Out.dataPath = outPath;
end

% NLSA output
if isfield( In, 'nlsaPath' )
    nlsaPath = In.nlsaPath;
else
    nlsaPath = fullfile( pwd, 'data/nlsa' ); % nlsa output
    In.nlsaPath = nlsaPath;
end

%% ABBREVIATED SOURCE AND TARGET COMPONENT NAMES
componentNames = {};
if isfield( In, 'sourceComponentName' )
    componentNames = [ componentNames ...
                       'sourceComponentName' In.sourceComponentName ];
end
if isfield( In, 'targetComponentName' )
    componentNames = [ componentNames ...
                       'targetComponentName' In.targetComponentName ];
end


%% DELAY-EMBEDDING ORIGINGS
In.nC  = numel( In.Src ); % number of source components
In.nCT = numel( In.Trg ); % number of target compoents

% Maximum number of delay embedding lags, sample left out in the 
% beginning/end of the analysis interval for source data
In.nE  = In.Src( 1 ).idxE( end ); 
In.nXB = In.Src( 1 ).nXB; 
In.nXA = In.Src( 1 ).nXA;
for iC = 2 : In.nC
    In.nE = max( In.nE, In.Src( iC ).idxE( end ) );
    In.nXB = max( In.nXB, In.Src( iC ).nXB );
    In.nXA = max( In.nXA, In.Src( iC ).nXA );
end

% Maximum number of delay embedding lags, sample left out in the 
% beginning/end of the analysis interval for targe data
nETMin  = In.Trg( 1 ).idxE( end ); % minimum number of delays for target data
In.nET  = In.Trg( 1 ).idxE( end ); % maximum number of delays for target data
In.nXBT = In.Trg( 1 ).nXB;
In.nXAT = In.Trg( 1 ).nXA;
for iC = 2 : In.nCT
    In.nET = max( In.nET, In.Trg( iC ).idxE( end ) );
    nETMin = min( In.nET, In.Trg( iC ).idxE( end ) );
    In.nXBT = min( In.nXBT, In.Trg( iC ).nXB );
    In.nXAT = min( In.nXAT, In.Trg( iC ).nXA );
end
nEMax = max( In.nE, In.nET );
nXBMax = max( In.nXB, In.nXBT );
nXAMax = max( In.nXA, In.nXAT );

%% NUMBER OF STAMPLES FOR IN-SAMPLE DATA
In.nR  = numel( In.Res ); % number of realizations, in-sample data
% Determine number of samples for in-sample data.
nSETot = 0;
idxT1 = zeros( 1, In.nR );
for iR = In.nR : -1 : 1
    % In.Res( iR ).nS:    number of samples
    % In.Res( iR ).idxT1: time origin for delay embedding
    % In.Res( iR ).nSE:   number of samples after delay embedding
    % In.Res( iR ).nSRec: number of samples for reconstruction
    In.Res( iR ).nS = In.Res( iR ).nSProd + nEXMax - 1 + nXBMax + nXAMax; 
    In.Res( iR ).idxT1 = nEMax + nXBMax;      
    In.Res( iR ).nSE = In.Res( iR ).nS - In.Res( iR ).idxT1 + 1 - nXAMax; 
    nSETot = nSETot + In.Res( iR ).nSE;
    In.Res( iR ).nSRec = In.Res( iR ).nSE + nETMin - 1; 
    idxT1( iR ) = In.Res( iR ).idxT1;
end
if In.nN == 0
   In.nN = nSETot;
end 
if In.nNS == 0
    In.nNS = nSETot;
end

%% OUT-OF-SAMPLE PARAMETER VALUES INHERITED FROM IN-SAMPLE DATA
if ifOse
    Out.nC           = In.nC;  % number of source components
    Out.nCT          = In.nCT; % number of target components
    Out.Src          = In.Src; % source component specification
    Out.Trg          = In.Trg; % target component specification
    Out.nE           = In.nE;  % number of delays for source data
    Out.nET          = In.nET; % number of delays for target data
    Out.nXB          = In.nXB; % left-out source samples before main interval
    Out.nXA          = In.nXA; % left-out source samples after main interval
    Out.nXBT         = In.nXBT; % left-out target samples before main interval
    Out.nXAT         = In.nXAT; % left-out target samples after main interval 
    Out.lDist        = In.lDist; % local distance function
    Out.tol          = In.tol; % cone kernel tolerance
    Out.zeta         = In.zeta; % cone kernel parameter zeta 
    Out.coneAlpha    = In.coneAlpha; % cone kernel parameter alpha 
    Out.diffOpType   = In.diffOpType; % diffusion operator type
    Out.alpha        = In.alpha; % diffusion maps parameter alpha
    Out.nN           = In.nN; % nearest neighbors for OSE pairwise distance 
    Out.nNO          = Out.nN; % nearest neighbors for OSE diffusion operator
    Out.epsilon      = 1; % Bandwidth parameter
    Out.nPhi         = In.nPhi; % diffusion eigenfunctions to compute
    Out.nNO          = In.nN; % number of nearest neighbors for OSE 
    Out.idxPhiRecOSE = In.idxPhiRec; % eigenfunctions to reconstruct
end

if ifOse && ifDen
    Out.denType      = In.denType; % density estimation type
    Out.denLDist     = In.denLDist; % local distance for density estimation
    Out.denZeta      = In.denZeta; % cone kernel parameter zeta
    Out.denConeAlpha = In.denConeAlpha; % cone kernel paramter alpha
    Out.denNN        = In.denNN; % nearest neighbors for KDE
    Out.denND        = In.denND; % manifold dimension for density estimation
    Out.denEpsilon   = In.denEpsilon; % bandwidth parmeter for KDE
end


%% NUMBER OF SAMPLES AND TIMESTAMPS FOR OUT-OF-SAMPLE DATA
    Out.nR  = numel( Out.Res ); % number of realizations, out-of-sample data
    idxT1O = zeros( 1, Out.nR );
    % Determine number of samples for out-of-sample data.
    for iR = Out.nR : -1 : 1
        % Out.Res( iR ).nS:    number of samples
        % Out.Res( iR ).idxT1: time origin for delay embedding
        % Out.Res( iR ).nSE:   number of samples after delay embedding
        % Out.Res( iR ).nSRec: number of samples for reconstruction
        Out.Res( iR ).nS = Out.Res( iR ).nSProd + nEXMax - 1 + nXBMax + nXAMax; 
        Out.Res( iR ).idxT1 = nEMax + nXBMax; 
        Out.Res( iR ).nSE = Out.Res( iR ).nS - Out.Res( iR ).idxT1 + 1-nXAMax; 
        Out.Res( iR ).nSERec = Out.Res( iR ).nSE + nETMin - 1; 
        idxT1O( iR ) = Out.Res( iR ).idxT1;
    end
end

%% IN-SAMPLE DATA COMPONENTS

% Loop over realizations for in-sample data
for iR = In.nR : -1 : 1
    
    % Realization tag                                
    tagR = [ 'beta'    num2str( In.Res( iR ).beta, '%1.3g' ) ...
             '_rho'    num2str( In.Res( iR ).rho, '%1.3g' ) ...
             '_sigma'  num2str( In.Res( iR ).sigma, '%1.3g' ) ...
             '_dt'     num2str( In.dt, '%1.3g' ) ...
             '_x0'     sprintf( '_%1.3g', In.Res( iR ).x0 ) ...
             '_nS'     int2str( In.Res( iR ).nS ) ...
             '_nSSpin' int2str( In.Res( iR ).nSSpin ) ...
             '_relTol' num2str( In.Res( iR ).relTol, '%1.3g' ) ...
             '_ifCent' int2str( In.Res( iR ).ifCent ) ];


    % Path to current realization
    pathR = fullfile( inPath,  tagR );

    % Source data assumed to be stored in a single batch
    partition = nlsaPartition( 'nSample', In.Res( iR ).nS ); 
    embPartition( iR ) = nlsaPartition( 'nSample', In.Res( iR ).nSE, ...
                                        'nBatch',  In.Res( iR ).nB  );
    recPartition( iR ) = nlsaPartition( 'nSample', In.Res( iR ).nSRec, ...
                                        'nBatch',  In.Res( iR ).nBRec );

    % Loop over source components
    for iC = In.nC : -1 : 1
   
        % Component tag
        tagC = [ 'idxX' sprintf( '_%i', In.Src( iC ).idxX ) ];

        % Component dimension
        nD = numel( In.Src( iC ).idxX );

        % Input file
        if nD == 3 && all( In.Src( iC ).idxX == [ 1 2 3 ] )
            fileName = 'dataX.mat';
        else
            fileName = [ 'dataX_' tagC '.mat' ];
        end

        % Filename for source data
        fList = nlsaFilelist( 'file', fileName ); 

        % Create source component                                           
        srcComponent( iC, iR ) = nlsaComponent( ...
                                    'partition',      partition, ...
                                    'dimension',      nD, ...
                                    'path',           pathR, ...
                                    'file',           fList, ...
                                    'componentTag',   tagC, ...
                                    'realizationTag', tagR  );

    end

    % Loop over target components 
    for iC = In.nCT : -1 : 1

        % Component tag
        tagC = [ 'idxX' sprintf( '_%i', In.Trg( iC ).idxX ) ];

        % Component dimension
        nD = numel( In.Trg( iC ).idxX );

        % Input file
        if nD == 3 && all( In.Trg( iC ).idxX == [ 1 2 3 ] )
            fileName = 'dataX.mat';
        else
            fileName = [ 'dataX_' tagC '.mat' ];
        end

        % Filename for target data
        fList = nlsaFilelist( 'file', fileName ); 

        % Create target component
        trgComponent( iC, iR ) = nlsaComponent( ...
                                    'partition',      partition, ...
                                    'dimension',      nD, ...
                                    'path',           pathC, ...
                                    'file',           fList, ...
                                    'componentTag',   tagC, ...
                                    'realizationTag', tagR  );
    end

end

% Loop over source components to create embedding templates
for iC = In.nC : -1 : 1
    switch In.Src( iC ).embFormat
        case 'evector'
            if In.Src( iC ).fdOrder <= 0
                embComponent( iC, 1 ) = nlsaEmbeddedComponent( ...
                                    'idxE',    In.Src( iC ).idxE, ... 
                                    'nXB',     In.Src( iC ).nXB, ...
                                    'nXA',     In.Src( iC ).nXA );
            else
                embComponent( iC, 1 ) = nlsaEmbeddedComponent_xi_e( ...
                                    'idxE',    In.Src( iC ).idxE, ... 
                                    'nXB',     In.Src( iC ).nXB, ...
                                    'nXA',     In.Src( iC ).nXA, ...
                                    'fdOrder', In.Src( iC ).fdOrder, ...
                                    'fdType',  In.Src( iC ).fdType );
            end
        case 'overlap'
            if In.Src( iC ).fdOrder <= 0
                embComponent( iC, 1 ) = nlsaEmbeddedComponent_o( ...
                                    'idxE',    In.Src( iC ).idxE, ...
                                    'nXB',     In.Src( iC ).nXB, ...
                                    'nXA',     In.Src( iC ).nXA );
            else
                embComponent( iC, 1 ) = nlsaEmbeddedComponent_xi_o( ...
                                    'idxE',    In.Src( iC ).idxE, ...
                                    'nXB',     In.Src( iC ).nXB, ...
                                    'nXA',     In.Src( iC ).nXA, ...
                                    'fdOrder', In.Src( iC ).fdOrder, ...
                                    'fdType',  In.Src( iC ).fdType );
            end
    end
end

% Loop over target components to create embedding templates
for iC = In.nCT : -1 : 1
    switch In.Trg( iC ).embFormat
        case 'evector'
            if In.Trg( iC ).fdOrder <= 0
                trgEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_e( ...
                                      'idxE',    In.Trg( iC ).idxE, ... 
                                      'nXB',     In.Trg( iC ).nXB, ...
                                      'nXA',     In.Trg( iC ).nXA );
            else
                trgEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_xi_e( ...
                                      'idxE',    In.Trg( iC ).idxE, ... 
                                      'nXB',     In.Trg( iC ).nXB, ...
                                      'nXA',     In.Trg( iC ).nXA, ...
                                      'fdOrder', In.Trg( iC ).fdOrder, ...
                                      'fdType',  In.Trg( iC ).fdType );
             end
        case 'overlap'
            if In.Trg( iC ).fdOrder <= 0 
                trgEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_o( ...
                                       'idxE',    In.Trg( iC ).idxE, ...
                                       'nXB',     In.Trg( iC ).nXB, ...
                                       'nXA',     In.Trg( iC ).nXA );
            else
                trgEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_xi_o( ...
                                       'idxE',    In.Trg( iC ).idxE, ...
                                       'nXB',     In.Trg( iC ).nXB, ...
                                       'nXA',     In.Trg( iC ).nXA, ...
                                       'fdOrder', In.Trg( iC ).fdOrder, ...
                                       'fdType',  In.Trg( iC ).fdType );
            end
    end
end


%% PROJECTED COMPONENTS
for iC = In.nCT : -1 : 1
    if isa( trgEmbComponent( iC, 1 ), 'nlsaEmbeddedComponent_xi' )
        prjComponent( iC ) = nlsaProjectedComponent_xi( ...
                             'nBasisFunction', In.nPhiPrj );
    else
        prjComponent( iC ) = nlsaProjectedComponent( ...
                             'nBasisFunction', In.nPhiPrj );
    end
end

%% OUT-OF-SAMPLE DATA COMPONENTS 
if ifOse
    for iR = Out.nR : -1 : 1

        % Realization tag                                
        tagR = [ 'beta'    num2str( Out.Res( iR ).beta, '%1.3g' ) ...
                 '_rho'    num2str( Out.Res( iR ).rho, '%1.3g' ) ...
                 '_sigma'  num2str( Out.Res( iR ).sigma, '%1.3g' ) ...
                 '_dt'     num2str( Out.dt, '%1.3g' ) ...
                 '_x0'     sprintf( '_%1.3g', Out.Res( iR ).x0 ) ...
                 '_nS'     int2str( Out.Res( iR ).nS ) ...
                 '_nSSpin' int2str( Out.Res( iR ).nSSpin ) ...
                 '_relTol' num2str( Out.Res( iR ).relTol, '%1.3g' ) ...
                 '_ifCent' int2str( Out.Res( iR ).ifCent ) ];

        % Path to current realization
        pathR = fullfile( outPath,  tagR );

        % Source data assumed to be stored in a single batch
        partition = nlsaPartition( 'nSample', Out.Res( iR ).nS ); 
        outEmbPartition( iR ) = nlsaPartition( ...
                                    'nSample', Out.Res( iR ).nSE, ...
                                    'nBatch',  Out.Res( iR ).nB  );
        oseRecPartition( iR ) = nlsaPartition( ...
                                    'nSample', Out.Res( iR ).nSERec, ...
                                    'nBatch',  Out.Res( iR ).nBRec ); 

        % Loop over out-of-sample source components
        for iC = Out.nC : -1 : 1

            % Component tag
            tagC = [ 'idxX' sprintf( '_%i', Out.Src( iC ).idxX ) ];

            % Component dimension
            nD = numel( Out.Src( iC ).idxX );

            % Input file
            if nD == 3 && all( Out.Src( iC ).idxX == [ 1 2 3 ] )
                fileName = 'dataX.mat';
            else
                fileName = [ 'dataX_' tagC '.mat' ];
            end

            % Filename for out-of-sample data
            fList = nlsaFilelist( 'file', fileName ); 

            outComponent( iC, iR ) = nlsaComponent( ...
                                        'partition',      partition, ...
                                        'dimension',      nD, ...
                                        'path',           pathR, ...
                                        'file',           fList, ...
                                        'componentTag',   tagC, ...
                                        'realizationTag', tagR  );
        end

        % Loop over out-of-sample target components
        for iC = Out.nCT : -1 : 1

            % Component tag
            tagC = [ 'idxX' sprintf( '_%i', Out.Trg( iC ).idxX ) ];

            % Component dimension
            nD = numel( Out.Trg( iC ).idxX );

            % Input file
            if nD == 3 && all( Out.Trg( iC ).idxX == [ 1 2 3 ] )
                fileName = 'dataX.mat';
            else
                fileName = [ 'dataX_' tagC '.mat' ];
            end

            % Filename for out-of-sample target data
            fList = nlsaFilelist( 'file', fileName ); 

            % Creat out-of-sample target component 
            outTrgComponent( iC, iR ) = nlsaComponent( ...
                                            'partition',      partition, ...
                                            'dimension',      nD, ...
                                            'path',           pathR, ...
                                            'file',           fList, ...
                                            'componentTag',   tagC, ...
                                            'realizationTag', tagR  );
        end
    end
       
    % Loop over out-of-sample source components to create embedding templates
    for iC = Out.nC : -1 : 1
        switch Out.Src( iC ).embFormat
            case 'evector'
                if Out.Src( iC ).fdOrder <= 0
                    outEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_e( ...
                                            'idxE',    Out.Src( iC ).idxE, ... 
                                            'nXB',     Out.Src( iC ).nXB, ...
                                            'nXA',     Out.Src( iC ).nXA );
                else
                    outEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_xi_e( ...
                                        'idxE',    Out.Src( iC ).idxE, ... 
                                        'nXB',     Out.Src( iC ).nXB, ...
                                        'nXA',     Out.Src( iC ).nXA, ...
                                        'fdOrder', Out.Src( iC ).fdOrder, ...
                                        'fdType',  Out.Src( iC ).fdType );
                end
            case 'overlap'
                if Out.Src( iC ).fdOrder <= 0
                    outEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_o( ...
                                            'idxE',    Out.Src( iC ).idxE, ...
                                            'nXB',     Out.Src( iC ).nXB, ...
                                            'nXA',     Out.Src( iC ).nXA );
                else

                    outEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_xi_o( ...
                                        'idxE',    Out.Src( iC ).idxE, ...
                                        'nXB',     Out.Src( iC ).nXB, ...
                                        'nXA',     Out.Src( iC ).nXA, ...
                                        'fdOrder', Out.Src( iC ).fdOrder, ...
                                        'fdType',  Out.Src( iC ).fdType );
                end
        end
    end

    
    % Loop over out-of-sample target components to create embedding templates
    for iC = Out.nCT : -1 : 1
        switch Out.Trg( iC ).embFormat
            case 'evector'
                if Out.Trg( iC ).fdOrder <= 0
                    outTrgEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_e( ...
                                      'idxE',    Out.Trg( iC ).idxE, ... 
                                      'nXB',     Out.Trg( iC ).nXB, ...
                                      'nXA',     Out.Trg( iC ).nXA );
                else
                    outTrgEmbComponent( iC, 1 ) ...
                                = nlsaEmbeddedComponent_xi_e( ...
                                      'idxE',    Out.Trg( iC ).idxE, ... 
                                      'nXB',     Out.Trg( iC ).nXB, ...
                                      'nXA',     Out.Trg( iC ).nXA, ...
                                      'fdOrder', Out.Trg( iC ).fdOrder, ...
                                      'fdType',  Out.Trg( iC ).fdType );
                 end
            case 'overlap'
                if Out.Trg( iC ).fdOrder <= 0
                    outTrgEmbComponent( iC, 1 ) = nlsaEmbeddedComponent_o( ...
                                       'idxE',    Out.Trg( iC ).idxE, ...
                                       'nXB',     Out.Trg( iC ).nXB, ...
                                       'nXA',     Out.Trg( iC ).nXA );
                else
                    outTrgEmbComponent( iC, 1 ) = ...
                                nlsaEmbeddedComponent_xi_o( ...
                                       'idxE',    Out.Trg( iC ).idxE, ...
                                       'nXB',     Out.Trg( iC ).nXB, ...
                                       'nXA',     Out.Trg( iC ).nXA, ...
                                       'fdOrder', Out.Trg( iC ).fdOrder, ...
                                       'fdType',  Out.Trg( iC ).fdType );
                end
        end
    end
end

% Select mode for pairwise distances based on embeddding format
if all( strcmp( { In.Src.embFormat }, 'overlap' ) )
    modeStr = 'implicit';
else
    modeStr = 'explicit';
end

%% PAIRWISE DISTANCE FOR DENSITY ESTIMATION FOR IN-SAMPLE DATA
if ifDen
    switch In.denLDist
        case 'l2' % L^2 distance
            denLDist = nlsaLocalDistance_l2( 'mode', modeStr );

        case 'at' % "autotuning" NLSA kernel
            denLDist = nlsaLocalDistance_at( 'mode', modeStr );

        case 'cone' % cone kernel
            denLDist = nlsaLocalDistance_cone( 'mode',      modeStr, ...
                                               'zeta',      In.denZeta, ...
                                               'tolerance', In.tol, ...
                                               'alpha',     In.denConeAlpha );
    end

    denDFunc = nlsaLocalDistanceFunction( 'localDistance', denLDist );

    denPDist = nlsaPairwiseDistance( 'nearestNeighbors', In.nN, ...
                                     'distanceFunction', denDFunc );

    denPDist = repmat( denPDist, [ In.nC 1 ] );
end

%% PAIRWISE DISTANCE FOR DENSITY ESTIMATION FOR OUT-OF-SAMPLE
if ifDen && ifOse
    switch Out.denLDist
        case 'l2' % L^2 distance
            denLDist = nlsaLocalDistance_l2( 'mode', modeStr );

        case 'at' % "autotuning" NLSA kernel
            denLDist = nlsaLocalDistance_at( 'mode', modeStr );

        case 'cone' % cone kernel
            denLDist = nlsaLocalDistance_cone( 'mode',      modeStr, ...
                                               'zeta',      Out.denZeta, ...
                                               'tolerance', Out.tol, ...
                                               'alpha',     Out.denConeAlpha );
    end

    denDFunc = nlsaLocalDistanceFunction( 'localDistance', denLDist );

    oseDenPDist = nlsaPairwiseDistance( 'nearestNeighbors', Out.nN, ...
                                        'distanceFunction', denDFunc );

    oseDenPDist = repmat( oseDenPDist, [ Out.nC 1 ] );
end

%% KERNEL DENSITY ESTIMATION FOR IN-SAMPLE DATA
if ifDen 
    switch In.denType
        case 'fb' % fixed bandwidth
            den = nlsaKernelDensity_fb( ...
                     'dimension',              In.denND, ...
                     'epsilon',                In.denEpsilon, ...
                     'bandwidthBase',          In.denEpsilonB, ...
                     'bandwidthExponentLimit', In.denEpsilonE, ...
                     'nBandwidth',             In.denNEpsilon );

        case 'vb' % variable bandwidth 
            den = nlsaKernelDensity_vb( ...
                     'dimension',              In.denND, ...
                     'epsilon',                In.denEpsilon, ...
                     'kNN',                    In.denNN, ...
                     'bandwidthBase',          In.denEpsilonB, ...
                     'bandwidthExponentLimit', In.denEpsilonE, ...
                     'nBandwidth',             In.denNEpsilon );
    end

    den = repmat( den, [ In.nC 1 ] );
end

%% KERNEL DENSITY ESTIMATION FOR OUT-OF-SAMPLE DATA
if ifDen && ifOse
    switch Out.denType
        case 'fb' % fixed bandwidth
            oseDen = nlsaKernelDensity_ose_fb( ...
                     'dimension',              Out.denND, ...
                     'epsilon',                Out.denEpsilon );

        case 'vb' % variable bandwidth 
            oseDen = nlsaKernelDensity_ose_vb( ...
                     'dimension',              Out.denND, ...
                     'kNN',                    Out.denNN, ...
                     'epsilon',                Out.denEpsilon );
    end


    oseDen = repmat( oseDen, [ Out.nC 1 ] );
end

%% PAIRWISE DISTANCES FOR IN-SAMPLE DATA
switch In.lDist
    case 'l2' % L^2 distance
        lDist = nlsaLocalDistance_l2( 'mode', modeStr );

    case 'at' % "autotuning" NLSA kernel
        lDist = nlsaLocalDistance_at( 'mode', modeStr ); 

    case 'cone' % cone kernel
        lDist = nlsaLocalDistance_cone( 'mode', modeStr, ...
                                        'zeta', In.zeta, ...
                                        'tolerance', In.tol, ...
                                        'alpha', In.coneAlpha );
end
if ifDen
    lScl = nlsaLocalScaling_pwr( 'pwr', 1 / In.denND );
    dFunc = nlsaLocalDistanceFunction_scl( 'localDistance', lDist, ...
                                           'localScaling', lScl );
else
    dFunc = nlsaLocalDistanceFunction( 'localDistance', lDist );
end
pDist = nlsaPairwiseDistance( 'distanceFunction', dFunc, ...
                              'nearestNeighbors', In.nN );

%% PAIRWISE DISTANCES FOR OUT-OF-SAMPLE DATA
if ifOse
    switch Out.lDist
        case 'l2' % L^2 distance
            lDist = nlsaLocalDistance_l2( 'mode', modeStr );

        case 'at' % "autotuning" NLSA kernel
            lDist = nlsaLocalDistance_at( 'mode', modeStr ); 

        case 'cone' % cone kernel
            lDist = nlsaLocalDistance_cone( 'mode', modeStr, ... 
                                            'zeta', In.zeta, ...
                                            'tolerance', In.tol, ...
                                            'alpha', In.coneAlpha );
    end

    lScl = nlsaLocalScaling_pwr( 'pwr', 1 / Out.denND );
    oseDFunc = nlsaLocalDistanceFunction_scl( 'localDistance', lDist, ...
                                              'localScaling', lScl );
    osePDist = nlsaPairwiseDistance( 'distanceFunction', oseDFunc, ...
                                     'nearestNeighbors', Out.nN );
end

%% SYMMETRIZED PAIRWISE DISTANCES
sDist = nlsaSymmetricDistance_gl( 'nearestNeighbors', In.nNS );

%% DIFFUSION OPERATOR FOR IN-SAMPLE DATA
switch In.diffOpType
    % global storage format, fixed bandwidth
    case 'gl'
        diffOp = nlsaDiffusionOperator_gl( 'alpha',          In.alpha, ...
                                           'epsilon',        In.epsilon, ...
                                           'nEigenfunction', In.nPhi );

    % global storage format, multiple bandwidth (automatic bandwidth selection)
    case 'gl_mb' 
        diffOp = nlsaDiffusionOperator_gl_mb( ...
                     'alpha',                  In.alpha, ...
                     'epsilon',                In.epsilon, ...
                     'nEigenfunction',         In.nPhi, ...
                     'bandwidthBase',          In.epsilonB, ...
                     'bandwidthExponentLimit', In.epsilonE, ...
                     'nBandwidth',             In.nEpsilon );

    % global storage format, multiple bandwidth (automatic bandwidth selection and SVD)
    case 'gl_mb_svd' 
        diffOp = nlsaDiffusionOperator_gl_mb_svd( ...
                     'alpha',                  In.alpha, ...
                     'epsilon',                In.epsilon, ...
                     'nEigenfunction',         In.nPhi, ...
                     'bandwidthBase',          In.epsilonB, ...
                     'bandwidthExponentLimit', In.epsilonE, ...
                     'nBandwidth',             In.nEpsilon );

    case 'gl_mb_bs'
        diffOp = nlsaDiffusionOperator_gl_mb_bs( ...
                     'alpha',                  In.alpha, ...
                     'epsilon',                In.epsilon, ...
                     'nEigenfunction',         In.nPhi, ...
                     'bandwidthBase',          In.epsilonB, ...
                     'bandwidthExponentLimit', In.epsilonE, ...
                     'nBandwidth',             In.nEpsilon );

end

%% DIFFUSION OPERATOR FOR OUT-OF-SAMPLE DATA
if ifOse
    switch Out.diffOpType
        case 'gl_mb_svd'
            oseDiffOp = nlsaDiffusionOperator_ose_svd( ...
                                       'alpha',          Out.alpha, ...
                                       'epsilon',        Out.epsilon, ...
                                       'epsilonT',       In.epsilon, ...
                                       'nNeighbors',     Out.nNO, ...
                                       'nNeighborsT',    In.nNS, ...
                                       'nEigenfunction', Out.nPhi );
        case 'gl_mb_bs'
            oseDiffOp = nlsaDiffusionOperator_ose_bs( ...
                                       'alpha',          Out.alpha, ...
                                       'epsilon',        Out.epsilon, ...
                                       'epsilonT',       In.epsilon, ...
                                       'nNeighbors',     Out.nNO, ...
                                       'nNeighborsT',    In.nNS, ...
                                       'nEigenfunction', Out.nPhi );

        otherwise
            oseDiffOp = nlsaDiffusionOperator_ose( ...
                                       'alpha',          Out.alpha, ...
                                       'epsilon',        Out.epsilon, ...
                                       'epsilonT',       In.epsilon, ...
                                       'nNeighbors',     Out.nNO, ...
                                       'nNeighborsT',    In.nNS, ...
                                       'nEigenfunction', Out.nPhi );
    end
end
 

%% LINEAR MAP FOR SVD OF TARGET DATA
linMap = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhiSVD );


%% RECONSTRUCTED TARGET COMPONENTS -- IN-SAMPLE DATA
% Reconstructed data from diffusion eigenfnunctions
recComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxPhiRec );

% Reconstructed data from SVD 
svdRecComponent = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxVTRec );


%% RECONSTRUCTED TARGET COMPONENTS -- OUT-OF-SAMPLE DATA
if ifOse
    % Nystrom extension
    oseEmbTemplate = nlsaEmbeddedComponent_ose_n( ...
        'eigenfunctionIdx', Out.idxPhiRecOSE );
    oseRecComponent = nlsaComponent_rec();
end

%% BUILD NLSA MODEL    

if ifOse
    if ifDen
        % Variable-bandwidth kernel and out-of-sample extension
        model = nlsaModel_den_ose( ...
                   'path',                            nlsaPath, ...
                   'sourceComponent',                 srcComponent, ...
                   'targetComponent',                 trgComponent, ...
                   componentNames{ : }, ...
                   'embeddingOrigin',                 idxT1, ...
                   'embeddingTemplate',               embComponent, ...
                   'targetEmbeddingTemplate',         trgEmbComponent, ...
                   'embeddingPartition',              embPartition, ...
                   'denPairwiseDistanceTemplate',     denPDist, ...
                   'kernelDensityTemplate',           den, ...
                   'pairwiseDistanceTemplate',        pDist, ...
                   'symmetricDistanceTemplate',       sDist, ...
                   'diffusionOperatorTemplate',       diffOp, ...
                   'projectionTemplate',              prjComponent, ...
                   'reconstructionTemplate',          recComponent, ...
                   'reconstructionPartition',         recPartition, ...
                   'linearMapTemplate',               linMap, ...
                   'svdReconstructionTemplate',       svdRecComponent, ...
                   'outComponent',                    outComponent, ...
                   'outTargetComponent',              outTrgComponent, ...
                   'outEmbeddingOrigin',              idxT1O, ...
                   'outEmbeddingTemplate',            outEmbComponent, ...
                   'outEmbeddingPartition',           outEmbPartition, ... 
                   'osePairwiseDistanceTemplate',     osePDist, ...
                   'oseDenPairwiseDistanceTemplate',  oseDenPDist, ...
                   'oseKernelDensityTemplate',        oseDen, ...
                   'oseDiffusionOperatorTemplate',    oseDiffOp, ...
                   'oseEmbeddingTemplate',            oseEmbTemplate, ...
                   'oseReconstructionPartition',      oseRecPartition );

    else
        % Out-of-sample extension
        model = nlsaModel_ose( ...
                   'path',                            nlsaPath, ...
                   'sourceComponent',                 srcComponent, ...
                   'targetComponent',                 trgComponent, ...
                   componentNames{ : }, ...
                   'embeddingOrigin',                 idxT1, ...
                   'embeddingTemplate',               embComponent, ...
                   'targetEmbeddingTemplate',         trgEmbComponent, ...
                   'embeddingPartition',              embPartition, ...
                   'pairwiseDistanceTemplate',        pDist, ...
                   'symmetricDistanceTemplate',       sDist, ...
                   'diffusionOperatorTemplate',       diffOp, ...
                   'projectionTemplate',              prjComponent, ...
                   'reconstructionTemplate',          recComponent, ...
                   'reconstructionPartition',         recPartition, ...
                   'linearMapTemplate',               linMap, ...
                   'svdReconstructionTemplate',       svdRecComponent, ...
                   'outComponent',                    outComponent, ...
                   'outTargetComponent',              outTrgComponent, ...
                   'outEmbeddingOrigin',              idxT1O, ...
                   'outEmbeddingTemplate',            outEmbComponent, ...
                   'outEmbeddingPartition',           outEmbPartition, ... 
                   'osePairwiseDistanceTemplate',     osePDist, ...
                   'oseDiffusionOperatorTemplate',    oseDiffOp, ...
                   'oseEmbeddingTemplate',            oseEmbTemplate, ...
                   'oseReconstructionPartition',      oseRecPartition );

    end
else 

    if ifDen                
        % Variable-bandwidth kernel 
        model = nlsaModel_den( ...
                   'path',                            nlsaPath, ...
                   'sourceComponent',                 srcComponent, ...
                   'targetComponent',                 trgComponent, ...
                   componentNames{ : }, ...
                   'embeddingOrigin',                 idxT1, ...
                   'embeddingTemplate',               embComponent, ...
                   'targetEmbeddingTemplate',         trgEmbComponent, ...
                   'embeddingPartition',              embPartition, ...
                   'denPairwiseDistanceTemplate',     denPDist, ...
                   'kernelDensityTemplate',           den, ...
                   'pairwiseDistanceTemplate',        pDist, ...
                   'symmetricDistanceTemplate',       sDist, ...
                   'diffusionOperatorTemplate',       diffOp, ...
                   'projectionTemplate',              prjComponent, ...
                   'reconstructionTemplate',          recComponent, ...
                   'reconstructionPartition',         recPartition, ...
                   'linearMapTemplate',               linMap, ...
                   'svdReconstructionTemplate',       svdRecComponent );

    else
        %  Basic NLSA model
        model = nlsaModel( ...
                   'path',                            nlsaPath, ...
                   'sourceComponent',                 srcComponent, ...
                   'targetComponent',                 trgComponent, ...
                   componentNames{ : }, ...
                   'embeddingOrigin',                 idxT1, ...
                   'embeddingTemplate',               embComponent, ...
                   'targetEmbeddingTemplate',         trgEmbComponent, ...
                   'embeddingPartition',              embPartition, ...
                   'pairwiseDistanceTemplate',        pDist, ...
                   'symmetricDistanceTemplate',       sDist, ...
                   'diffusionOperatorTemplate',       diffOp, ...
                   'projectionTemplate',              prjComponent, ...
                   'reconstructionTemplate',          recComponent, ...
                   'reconstructionPartition',         recPartition, ...
                   'linearMapTemplate',               linMap, ...
                   'svdReconstructionTemplate',       svdRecComponent );
   end
end

