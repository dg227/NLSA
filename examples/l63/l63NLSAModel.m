function [ model, In, Out ] = l63NLSAModel( In, Out )
% L63NLSAModel Build NLSA model for Lorenz 63 system
%
%  [ model, In ] = l63NLSAModel( In ) builds an NLSA model based on the
%  parameters specified in the structure In. 

%  [ model, In, Out ] = l63NLSAModel( In, Out ) builds an NLSA model
%  with support for out-of-sample (test) data. The model parameters for the
%  in-sample (training) and out-of-sample (test) data are specified in the 
%  structures In and Out, respectively. 
%
%  l63NLSAModel uses the parameter values in In and Out to create arrays of 
%  nlsaComponent objects representing the in-sample and out-of-sample data, 
%  respectively. Then, it passses these arrays along with the parameter values 
%  to function /utils/nlsaModelFromPars. The latter, prepares appropriate 
%  arguments for the nlsaModel class constructors, and then calls the 
%  constructors to build the model.
%
%  l63NLSAModel is meant to be called by higher-level functions tailored to 
%  specific data analysis/forecasting experiments. 
%
%  For additional information on the class constructors see 
%
%      /nlsa/classes/nlsaModel_base/nlsaModel_base.m
%      /nlsa/classes/nlsaModel/nlsaModel.m
%      /nlsa/classes/nlsaModel_den/nlsaModel_den.m
%      /nlsa/classes/nlsaModel_den_ose/nlsaModel_den_ose.m
%
% Modified 2020/06/06
 

%% PRELIMINARY CHECKS
% Check if we are doing out-of-sample extension
if ~exist( 'Out', 'var' )
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
if ifOse && isfield( Out, 'Trg' )
    ifOutTrg = true; % Include out-of-sample target data
else
    ifOutTrg = false;
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
    In.Res( iR ).nS = In.Res( iR ).nSProd + nEMax - 1 + nXBMax + nXAMax; 
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
    Out.dt           = In.dt;  % sampling interval
    Out.nC           = In.nC;  % number of source components
    Out.Src          = In.Src; % source component specification
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

if ifOse && ifOutTrg
    Out.nCT          = In.nCT; % number of target components
end

if ifOse && ifDen
    Out.denType      = In.denType; % density estimation type
    Out.denLDist     = In.denLDist; % local distance for density estimation
    Out.denZeta      = In.denZeta; % cone kernel parameter zeta
    Out.denConeAlpha = In.denConeAlpha; % cone kernel paramter alpha
    Out.denNN        = In.denNN; % nearest neighbors for KDE
    Out.denND        = In.denND; % manifold dimension for density estimation
    Out.denEpsilon   = 1; % bandwidth parmeter for KDE
end


%% NUMBER OF SAMPLES AND TIMESTAMPS FOR OUT-OF-SAMPLE DATA
if ifOse
    Out.nR  = numel( Out.Res ); % number of realizations, out-of-sample data
    idxT1O = zeros( 1, Out.nR );
    % Determine number of samples for out-of-sample data.
    for iR = Out.nR : -1 : 1
        % Out.Res( iR ).nS:    number of samples
        % Out.Res( iR ).idxT1: time origin for delay embedding
        % Out.Res( iR ).nSE:   number of samples after delay embedding
        % Out.Res( iR ).nSRec: number of samples for reconstruction
        Out.Res( iR ).nS = Out.Res( iR ).nSProd + nEMax - 1 + nXBMax + nXAMax; 
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
                                    'path',           pathR, ...
                                    'file',           fList, ...
                                    'componentTag',   tagC, ...
                                    'realizationTag', tagR  );
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

        if ifOutTrg

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
    end
end
       

%% ASSEMBLE DATA COMPONENTS AND CALL FUNCTION TO BUILD THE MODEL
Data.src = srcComponent;
Data.trg = trgComponent;
if ifOse
    Data.out = outComponent;
end
if ifOutTrg
    Data.outTrg = outTrgComponent;
end
Pars.In = In;
if ifOse
    Pars.Out = Out;
end

model = nlsaModelFromPars( Data, Pars );
