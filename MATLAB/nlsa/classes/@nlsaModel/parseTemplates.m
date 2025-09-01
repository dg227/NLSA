function constrArgs = parseTemplates( varargin ) 
%PARSETEMPLATES Construct property name-value pairs for the nlsaModel class
%   constructor from templates.
%
%   The input arguments of parseTemplates are passed as name-value pairs using
%   the syntax:
%
%   constrArgs = parseTemplates( propName1, propVal1, propName2, propVal2, ... ).
%
%   The following input names can be specified in addition to those in the 
%   parseTemplates method of the parent nlsaModel_base class:
%
%   'pairwiseDistanceTemplate': An nlsaPairwiseDistance object specifying
%      the method and parameters (e.g., number of nearest neighbors) used 
%      to compute the graph edge weights (pairwise distances) for the dataset.
%      If 'pairwiseDistanceTemplate' is not specified, it is set to the L2
%      distance, and the number of nearest neighnors nN to 1/10 of the number
%      of samples after delay embedding.
%
%   'symmetricDistanceTemplate': An nlsaSymmetricDistance object 
%      implementing the pairwise distance symmetrization. The number of 
%      nearest neighbors nNS in this object cannot exceed nN. If 
%      'symmetricDistanceTemplate' is not specified, nNS is set equal to nN.
%
%   'diffusionOperatorTemplate': An nlsaDiffusionOperator object specifying 
%      the diffusion operator (Laplacian) in the data analysis model. 
%
%   'koopmanOperatorTemplate': An nlsaKoopmanOperator object specifying the
%      Koopman operator. If koopmanOperatorTemplate is not specified, the
%      koopmanOperator property of the constructed model is set to empty. 
% 
%   'projectionTemplate': An array of nlsaProjection objects specifying the
%      projections of the target data in delay-embedding space onto the
%      diffusion eigenfunctions. 'projectionTemplate' must be a vector of size 
%      [ nCT 1 ], where nCT is the number of target components.   
%
%   'reconstructionPartition': An [ 1 nR ]-sized vector of nlsaPartition objects
%      specifying how each realization of the reconstructed data is to be 
%      partitioned. That is, in the resulting nlsaModel_base object, the
%      properties recComponent( iC, iR ).partition and 
%      svdRecComponent( iC, iR ).partition are both set to partition( iR ) 
%      for all iC and iR. The number of samples in partition( iR ) 
%      must not exceed the number of samples in the iR-th realization of the 
%      delay-embedded data, plus nE( iCT ) - 1, where nE( iCT ) is the number
%      of delays for target component iCT. If 'reconstructionPartition' is 
%      not specified, the partition is set to a single-batch partition with
%      the maximum number of samples allowed.
% 
%   'reconstructionTemplate': An array of nlsaComponent_rec_phi objects 
%      specifying the reconstruction of the eigenfunction-projected target
%      data. 'reconstructionTemplate' must be either a scalar or a vector of
%      size [ 1 nR ], where nR is the number of realizations. 
%
%   'koopmanProjectionTemplate': An array of nlsaProjection objects 
%      specifying the projections of the target data in delay-embedding space 
%      onto the Koopman eigenfunctions. 'projectionTemplate' must be a vector 
%      of size [ nCT 1 ], where nCT is the number of target components.   
%
%   'koopmanReconstructionTemplate': Similar to 'reconstructionTemplate', but
%      for reconstruction using Koopman eigenfunctions.
%
%   'linearMapTemplate': An nlsaLinearMap object implementing the SVD of the
%      projected data. 'linearMapTemplate' can be either a scalar or a vector; 
%      in the latter case the eigenfunctions of the linear maps must be nested
%      as described in the nlsaModel class constructor function.  
%       a temporal space and the target data. 
%
%   'svdReconstructionTemplate': An arry of nlsaComponent_rec_phi objects
%      specifying the reconstruction of the SVD modes. 
%      'svdReconstructionTemplate' can be a scalar, a vector of size [ 1 nA ], a
%      matrix of size [ nR nA ], or an array of size [ nCT nR nA ]. nA is the 
%      number of elements of the linear map vector.  
%
%   'sourceComponentName': A string which, if defined, is used to replace the
%      default directory name of the pairwise distance data. This option is
%      useful to avoid long directory names in datasets with several
%      components, but may lead to non-uniqueness of the filename structure
%      and overwriting of results. 
%
%   'sourceRealizationName': Similar to 'sourceComponentName', but used to
%      compress the realization-dependent part of the pairwise distance
%      directory.
% 
%   'targetComponentName', 'targetRealizationName': Same as 
%      'sourceComponentName' and 'sourceRealizationName', respectively, but
%      for the target data. 
%
%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2021/03/18 


%% CONSTRUCTOR PROPERTY LIST
propName = nlsaModel.listConstructorProperties; 
propVal  = cell( size( propName ) );
ifProp   = false( size( propName ) );
idxProp  = zeros( size( propName ) );
nProp    = numel( idxProp );
%% SUPERCLASS CONSTRUCTOR ARGUMENTS
parentConstrArgs = nlsaModel_base.parseTemplates( varargin{ : } );
iEmbComponentQ = [];
iEmbComponentT = [];
for iProp = 1 : 2 : numel( parentConstrArgs )
    switch parentConstrArgs{ iProp }
        case 'embComponent'
            iEmbComponent = iProp + 1;
        case 'embComponentQ'
            iEmbComponentQ = iProp + 1;
        case 'embComponentT'
            iEmbComponentT = iProp + 1;
        case 'trgComponent'
            iTrgComponent = iProp + 1;
        case 'trgEmbComponent'
            iTrgEmbComponent = iProp + 1;
    end
end
partition = getPartition( parentConstrArgs{ iEmbComponent }( 1, : ) );
if ~isempty( iEmbComponentT )
    partitionT = getPartition( parentConstrArgs{ iEmbComponentT }( 1 ) );
else
    partitionT = partition;
end
if ~isempty( iEmbComponentQ )
    partitionQ = getPartition( parentConstrArgs{ iEmbComponentQ }( 1 ) );
else
    partitionQ = partition;
end
nCT = size( parentConstrArgs{ iTrgEmbComponent }, 1 );
nR  = size( parentConstrArgs{ iEmbComponent }, 2 );

%% MODEL PATH
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'path' )
        if isSet == 1
            error( 'Path argument has been already specified' )
        end
        if ischar( varargin{ i + 1 } )
            modelPath = varargin{ i + 1 };
            isSet = 1;
        else
            error( 'Invalid path specification' )
        end
    end
end
if ~isSet
    modelPath = pwd;
end

%% PAIRWISE DISTANCE
% Parse distance template and set distance partition
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'pairwiseDistance' )
        iPDistance = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'pairwiseDistanceTemplate' )
        if ~isempty( propVal{ iPDistance } )
            error( 'A pairwise distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPairwiseDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iPDistance } = varargin{ i + 1 };
            ifProp( iPDistance )  = true;
        else
            error( 'The pairwise distance template must be specified as a scalar nlsaPairwiseDistance object' )
        end
    end
end
if isempty( propVal{ iPDistance } )
    propVal{ iPDistance } = nlsaPairwiseDistance( ...
      'nearestNeighbors', round( getNTotalSample( partition ) / 10 ) );
    ifProp( iPDistance )  = true;
end
nN = getNNeighbors( propVal{ iPDistance } );

propVal{ iPDistance } = setPartition( propVal{ iPDistance }, partitionQ );
propVal{ iPDistance } = setPartitionTest( propVal{ iPDistance }, partitionT );
tag = getTag( propVal{ iPDistance } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iPDistance } = setTag( propVal{ iPDistance }, ...
        [ tag getDefaultTag( propVal{ iPDistance } ) ] ); 

% Determine distance-specific directories
pth = concatenateTags( parentConstrArgs{ iEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sourceComponentName' ) 
        if ~isSet
            pth{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'sourceComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sourceRealizationName' )
        if ~isSet
            pth{ 2 } = varargin{ i + 1 };
            break
        else
            error( 'sourceRealizationName has been already set' )
        end
    end  
end
pth = strjoin_e( pth, '_' );
% Assign pairwise distance paths and filenames
modelPathD = fullfile( modelPath, 'processed_data', pth, ...
                       getTag( propVal{ iPDistance } )  );
propVal{ iPDistance } = setPath( propVal{ iPDistance }, modelPathD );
propVal{ iPDistance } = setDefaultSubpath( propVal{ iPDistance } ); 
propVal{ iPDistance } = setDefaultFile( propVal{ iPDistance } );
mkdir( propVal{ iPDistance } )


%% SYMMETRIC DISTANCE
% Parse symmetric distance template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'symmetricDistance' )
        iSDistance = iProp;
        break
    end  
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'symmetricDistanceTemplate' )
        if ~isempty( propVal{ iSDistance } )
            error( 'A symmetric distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaSymmetricDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iSDistance } = varargin{ i + 1 };
            ifProp( iSDistance ) = true;
        else
            error( 'The symmetric distance template must be specified as a scalar nlsaSymmetricDistance object' )
        end
    end
end

if isempty( propVal{ iSDistance } )
    propVal{ iSDistance } = nlsaSymmetricDistance_gl( 'nearestNeighbors', nN );
    ifProp( iSDistance ) = true;
end

if getNNeighbors( propVal{ iSDistance } ) > nN
    error( 'The number of nearest neighbors in the symmetric distance matrix cannot exceed the number of neareast neighbors in the pairwise (non-symmetric) distance matrix' )
end
propVal{ iSDistance } = setPartition( propVal{ iSDistance }, partitionQ );
tag = getTag( propVal{ iSDistance } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iSDistance } = setTag( propVal{ iSDistance }, ...
        [ tag getDefaultTag( propVal{ iSDistance } ) ] ); 

% Assign symmetric distance paths and filenames
modelPathS            = fullfile( modelPathD, getTag( propVal{ iSDistance } ) );
propVal{ iSDistance } = setPath( propVal{ iSDistance }, modelPathS );
propVal{ iSDistance } = setDefaultSubpath( propVal{ iSDistance } );
propVal{ iSDistance } = setDefaultFile( propVal{ iSDistance } );
mkdir( propVal{ iSDistance } )

%% DIFFUSION OPERATOR
% Parse diffusion operator template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'diffusionOperator' )
        iDiffOp = iProp;
        break
    end  
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'diffusionOperatorTemplate' )
        if ~isempty( propVal{ iDiffOp } )
            error( 'A diffusion operator template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaDiffusionOperator' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iDiffOp } = varargin{ i + 1 };
        else
            error( 'The diffusion operator template must be specified as a scalar nlsaDiffusionOperator object' )
        end
        ifProp( iDiffOp ) = true;
    end
end
if isempty( propVal{ iDiffOp } )
    propVal{ iDiffOp } = nlsaDiffusionOperator_gl();
    ifProp( iDiffOp ) = true;
end
nPhi   = getNEigenfunction( propVal{ iDiffOp } );
if nPhi > getNTotalSample( partition )
    msgStr = [ 'Number of diffusion eigenfunctions cannot exceed ' ...
               'the number of available samples.' ]; 
    error( msgStr )
end
propVal{ iDiffOp } = setPartition( propVal{ iDiffOp }, partition );
propVal{ iDiffOp } = setPartitionTest( propVal{ iDiffOp }, partition );
%propVal{ iDiffOp } = setPartitionTest( propVal{ iDiffOp }, partitionQ );
if isa( propVal{ iDiffOp }, 'nlsaDiffusionOperator_batch' )
    propVal{ iDiffOp } = setNNeighbors( ...
                            propVal{ iDiffOp }, ...
                            getNNeighborsMax( propVal{ iSDistance } ) );
end
tag = getTag( propVal{ iDiffOp } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iDiffOp } = setTag( propVal{ iDiffOp }, ...
        [ tag getDefaultTag( propVal{ iDiffOp } ) ] ); 

% Assign diffusion operator  paths and filenames
modelPathL = fullfile( modelPathS, getTag( propVal{ iDiffOp } ) );
propVal{ iDiffOp } = setDefaultSubpath( propVal{ iDiffOp } );
propVal{ iDiffOp } = setPath( propVal{ iDiffOp }, modelPathL );
mkdir( propVal{ iDiffOp } )
propVal{ iDiffOp } = setDefaultFile( propVal{ iDiffOp } );


%% PROJECTED DATA

% Parse projection templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'prjComponent' )
        iPrjComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'projectionTemplate' )
        if ~isempty( propVal{ iPrjComponent } )
            error( 'A projection template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
            && isscalar( varargin{ i + 1 } ) 
            propVal{ iPrjComponent } = repmat( varargin{ i + 1 }, [ nCT 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
           && isvector( varargin{ i + 1 } ) ...
           && numel( varargin{ i + 1 } ) == nCT
            propVal{ iPrjComponent } = varargin{ i + 1 };
            if isrow( propVal{ iPrjComponent } )
                propVal{ iPrjComponent } = propVal{ iPrjComponent }';
            end
        else
            error( 'The projection template must be specified as a scalar nlsaProjectedComponent object, or vector of nlsaProjectedComponent objects of size equal to the number of target components' )
        end
        ifProp( iPrjComponent ) = true;
    end
end
if isempty( propVal{ iPrjComponent } )
    for iC = nCT : -1 : 1
        propVal{ iPrjComponent }( iC ) = nlsaProjectedComponent( 'nBasisFunction', nPhi );
    end
    ifProp( iPrjComponent ) = true;
end

% Setup collective tags for projected data
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'projectionRealizationName' )
        if ~isSet
            pthR = varargin{ i + 1 };
            isSet = true;
            break
        else
            error( 'Projection realization name has been already specified' )
        end      
    end  
end

for iC = 1 : nCT
    pth = concatenateTags( parentConstrArgs{ iTrgEmbComponent }( iC, : ) );
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'targetRealizationName' )
            if ~isSet
                pth{ 2 } = varargin{ i + 1 };
                isSet = true;
                break
            else
                error( 'Target realization name has been already specified' )
            end      
        end  
    end
    pth = strjoin_e( pth, '_' );

    propVal{ iPrjComponent }( iC ) = setPartition( ...
        propVal{ iPrjComponent }( iC ), partition );
    propVal{ iPrjComponent }( iC ) = setPath( ...
        propVal{ iPrjComponent }( iC ), fullfile( modelPathL, pth ) );
    propVal{ iPrjComponent }( iC ) = setEmbeddingSpaceDimension( ...
       propVal{ iPrjComponent }( iC ), ...
       getEmbeddingSpaceDimension( ...
           parentConstrArgs{ iTrgEmbComponent }( iC, 1 ) ) );
    propVal{ iPrjComponent }( iC ) = setDefaultSubpath( ...
        propVal{ iPrjComponent }( iC ) );
    propVal{ iPrjComponent }( iC ) = setDefaultFile( ...
        propVal{ iPrjComponent }( iC ) );
    mkdir( propVal{ iPrjComponent }( iC ) )
end
if ~isCompatible( propVal{ iPrjComponent } )
    error( 'Incompatible projection components' )
end
if ~isCompatible( propVal{ iPrjComponent }, propVal{ iDiffOp } )
    error( 'Incompatible projection components and diffusion operator' )
end
mkdir( propVal{ iPrjComponent } )


%% RECONSTRUCTED COMPONENTS
% Parse reconstruction templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'recComponent' )
        iRecComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'reconstructionTemplate' )
        if ~isempty( propVal{ iRecComponent } )
            error( 'A reconstruction template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
            && isvector( varargin{ i + 1 } ) 
            nRecPhi = numel( varargin{ i + 1 } );
            propVal{ iRecComponent } = repmat( ...
                reshape( varargin{ i + 1 }, [ 1 1 nRecPhi ] ), [ nCT nR ] );
        else
            msgStr = [ 'The reconstruction template must be specified as ' ...
                       'a vector of nlsaComponent_rec_phi objects.' ];
            error( msgStr ) 
        end
        ifProp( iRecComponent ) = true;
    end
end
if isempty( propVal{ iRecComponent } )
    for iR = nR : -1 : 1
        for iC = nCT : -1 : 1
            propVal{ iRecComponent }( iC, iR ) = ...
                nlsaComponent_rec_phi( 'basisFunctionIdx', 1 );
        end
    end
    ifProp( iRecComponent ) = true;
end
nRecPhi = size( propVal{ iRecComponent }, 3 );

% Determine maximum number of samples in each realization of the reconstructed
% data
maxNSRR = zeros( 1, nR );
for iR = 1 : nR
    maxNSRR( iR ) = getNSample( ...
                      parentConstrArgs{ iTrgEmbComponent }( 1, iR )  )...
                  + getEmbeddingWindow( ...
                      parentConstrArgs{ iTrgEmbComponent }( 1, iR ) ) ...
                  - 1;
end

% Parse reconstruction partition templates
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'reconstructionPartition' )
        if isSet
            error( 'Partition templates for the reconstructed data have been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
           && isscalar( varargin{ i + 1 } )
            recPartition = repmat( varargin{ i + 1 }, [ 1 nR ] );
        elseif isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nR
            recPartition = varargin{ i + 1 };
        else
            error( 'Data partitions must be specified as a scalar nlsaPartition object, or a vector of nlsaPartition objects of size equal to the number of ensemble realizations' )
        end
        isSet = true;
    end
end
if ~isSet
    for iR = nR : -1 : 1
        recPartition( iR ) = nlsaPartition( 'nSample', maxNSRR( iR ) );
    end
end
for iR = 1 : nR
    if getNSample( recPartition( iR ) ) > maxNSRR( iR )
        error( 'Number of reconstructed samples is above maximum value' )
    end
end

% Setup reconstructed component partitions, tags, directories, and filenames
for iRec = 1 : nRecPhi
    for iR = 1 : nR
        for iC = 1 : nCT
            propVal{ iRecComponent }( iC, iR, iRec ) = setDimension( ...
               propVal{ iRecComponent }( iC, iR, iRec ), ...
               getDimension( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
            propVal{ iRecComponent }( iC, iR, iRec ) = setPartition( ...
                propVal{ iRecComponent }( iC, iR, iRec ), recPartition( iR ) );
            propVal{ iRecComponent }( iC, iR, iRec ) = setDefaultTag( ...
                propVal{ iRecComponent }( iC, iR, iRec ) );
            propVal{ iRecComponent }( iC, iR, iRec ) = setComponentTag( ...
                propVal{ iRecComponent }( iC, iR, iRec ), ...
                getComponentTag( ...
                    parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
            propVal{ iRecComponent }( iC, iR, iRec ) = setRealizationTag( ...
               propVal{ iRecComponent }( iC, iR, iRec ), ...
               getRealizationTag( ...
                   parentConstrArgs{ iTrgComponent }( iC, iR ) ) );    
            propVal{ iRecComponent }( iC, iR, iRec ) = setDefaultSubpath( ...
                propVal{ iRecComponent }( iC, iR, iRec ) );
            propVal{ iRecComponent }( iC, iR, iRec ) = setDefaultFile( ...
                propVal{ iRecComponent }( iC, iR, iRec ) );

            pth = strjoin_e( getTag( ...
                parentConstrArgs{ iTrgEmbComponent }( iC, iR ) ), '_' );   
            pth = fullfile( modelPathL, pth, ...
                getBasisFunctionTag( ...
                    propVal{ iRecComponent }( iC, iR, iRec ) ) );
            propVal{ iRecComponent }( iC, iR, iRec ) = ...
                setPath( propVal{ iRecComponent }( iC, iR, iRec ), pth );
        end
    end
end
mkdir( propVal{ iRecComponent } )

%% KOOPMAN OPERATOR
% Parse Koopman operator template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'koopmanOperator' )
        iKoopmanOp = iProp;
        break
    end  
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'koopmanOperatorTemplate' )
        if ~isempty( propVal{ iKoopmanOp } )
            error( 'A Koopman operator template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaKoopmanOperator' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iKoopmanOp } = varargin{ i + 1 };
        else
            msgStr = [ 'The Koopman operator template must be specified ' ...
                       'as a scalar nlsaKoopmanOperator object' ];
            error( msgStr )
        end
        ifProp( iKoopmanOp ) = true;
    end
end

if ifProp( iKoopmanOp )
    propVal{ iKoopmanOp } = setPartition( propVal{ iKoopmanOp }, partition );
    %propVal{ iKoopmanOp } = setPartitionTest( propVal{ iKoopmanOp }, ...
    %                            partitionQ );
    propVal{ iKoopmanOp } = setPartitionTest( propVal{ iKoopmanOp }, ...
                                partition );
    idxPhi = getBasisFunctionIndices( propVal{ iKoopmanOp } );
    if any( idxPhi > nPhi )
        msgStr = [ 'Diffusion eigenfunctions requested for Koopman ' ...
                   'operator approximation exceeds available ' ...
                   'eigenfunctions.' ];
        error( msgStr )
    end
    tag = getTag( propVal{ iKoopmanOp } );
    if ~isempty( tag )
        tag = [ tag '_' ];
    end
    propVal{ iKoopmanOp } = setTag( propVal{ iKoopmanOp }, ...
            [ tag getDefaultTag( propVal{ iKoopmanOp } ) ] ); 

    % Assign Koopman operator  paths and filenames
    modelPathK = fullfile( modelPathL, getTag( propVal{ iKoopmanOp } ) );
    propVal{ iKoopmanOp } = setDefaultSubpath( propVal{ iKoopmanOp } );
    propVal{ iKoopmanOp } = setPath( propVal{ iKoopmanOp }, modelPathK );
    mkdir( propVal{ iKoopmanOp } )
    propVal{ iKoopmanOp } = setDefaultFile( propVal{ iKoopmanOp } );
end


%% KOOPMAN-PROJECTED DATA

% Parse projection templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'koopmanPrjComponent' )
        iKoopmanPrjComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'koopmanProjectionTemplate' )
        if ~isempty( propVal{ iKoopmanPrjComponent } )
            error( 'A projection template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
            && isscalar( varargin{ i + 1 } ) 
            propVal{ iKoopmanPrjComponent } = repmat( varargin{ i + 1 }, ...
                                                      [ nCT 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
           && isvector( varargin{ i + 1 } ) ...
           && numel( varargin{ i + 1 } ) == nCT
            propVal{ iKoopmanPrjComponent } = varargin{ i + 1 };
            if isrow( propVal{ iKoopmanPrjComponent } )
                propVal{ iKoopmanPrjComponent } = ...
                    propVal{ iKoopmanPrjComponent }';
            end
        else
            error( 'The projection template must be specified as a scalar nlsaProjectedComponent object, or vector of nlsaProjectedComponent objects of size equal to the number of target components' )
        end
        ifProp( iKoopmanPrjComponent ) = true;
    end
end
if ifProp( iKoopmanPrjComponent ) && ~ifProp( iKoopmanOp ) 
    msgStr =[ 'Koopman projection templates cannot be specified without ' ...
              'spcifying a Koopman operator template.' ];  
    error( msgStr ) 
end

% We only assign Koopman-projected components if Koopman operator has been
% specified
if ifProp( iKoopmanOp )
    if isempty( propVal{ iKoopmanPrjComponent } )
        for iC = nCT : -1 : 1
            propVal{ iKoopmanPrjComponent }( iC ) = ...
                nlsaProjectedComponent( 'nBasisFunction', nPhi );
        end
        ifProp( iKoopmanPrjComponent ) = true;
    end

    % Setup collective tags for projected data
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'projectionRealizationName' )
            if ~isSet
                pthR = varargin{ i + 1 };
                isSet = true;
                break
            else
                msgStr = [ 'Projection realization name has been already ' ...
                           'specified.' ];
                error( msgStr  )
            end      
        end  
    end

    for iC = 1 : nCT
        pth = concatenateTags( parentConstrArgs{ iTrgEmbComponent }( iC, : ) );
        isSet = false;
        for i = 1 : 2 : nargin
            if strcmp( varargin{ i }, 'targetRealizationName' )
                if ~isSet
                    pth{ 2 } = varargin{ i + 1 };
                    isSet = true;
                    break
                else
                    msgStr = [ 'Target realization name has been already ' ...
                               'specified.' ];
                    error( msgStr  )
                end      
            end  
        end
        pth = strjoin_e( pth, '_' );

        propVal{ iKoopmanPrjComponent }( iC ) = setPartition( ...
            propVal{ iKoopmanPrjComponent }( iC ), partition );
        propVal{ iKoopmanPrjComponent }( iC ) = setPath( ...
            propVal{ iKoopmanPrjComponent }( iC ), ...
            fullfile( modelPathK, pth ) );
        propVal{ iKoopmanPrjComponent }( iC ) = setEmbeddingSpaceDimension( ...
           propVal{ iKoopmanPrjComponent }( iC ), ...
           getEmbeddingSpaceDimension( ...
               parentConstrArgs{ iTrgEmbComponent }( iC, 1 ) ) );
        propVal{ iKoopmanPrjComponent }( iC ) = setDefaultSubpath( ...
            propVal{ iKoopmanPrjComponent }( iC ) );
        propVal{ iKoopmanPrjComponent }( iC ) = setDefaultFile( ...
            propVal{ iKoopmanPrjComponent }( iC ) );
        mkdir( propVal{ iKoopmanPrjComponent }( iC ) )
    end
    if ~isCompatible( propVal{ iKoopmanPrjComponent } )
        error( 'Incompatible projection components' )
    end
    if ~isCompatible( propVal{ iKoopmanPrjComponent }, propVal{ iKoopmanOp } )
        error( 'Incompatible projection components and Koopman operator' )
    end
    mkdir( propVal{ iKoopmanPrjComponent } )
end


%% KOOPMAN-RECONSTRUCTED COMPONENTS
% Parse reconstruction templates
% We only assign Koopman-reconstructed components if Koopman operator has been
% specified
if ifProp( iKoopmanOp )
    for iProp = 1 : nProp 
        if strcmp( propName{ iProp }, 'koopmanRecComponent' )
            iKoopmanRecComponent = iProp;
            break
        end
    end
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'koopmanReconstructionTemplate' )
            if ~isempty( propVal{ iKoopmanRecComponent } )
                msgStr = [ 'A Koopman reconstruction template has been ' ...
                           'already specified.' ];
                error( msgStr  )
            end
            if isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
                && isvector( varargin{ i + 1 } ) 
                nRecKoop = numel( varargin{ i + 1 } );
                propVal{ iKoopmanRecComponent } = repmat( ... 
                    reshape( varargin{ i + 1 }, [ 1 1 nRecKoop ] ), ...
                             [ nCT nR ] );
            else

                msgStr = [ 'The Koopman reconstruction template must be ' ...
                           'specified as a vector of nlsaComponent_rec_phi ' ...
                           'objects.' ];
                error( msgStr ) 
            end
            ifProp( iKoopmanRecComponent ) = true;
        end
    end
    if isempty( propVal{ iKoopmanRecComponent } )
        for iR = nR : -1 : 1
            for iC = nCT : -1 : 1
                propVal{ iKoopmanRecComponent }( iC, iR ) = ...
                    nlsaComponent_rec_phi( 'basisFunctionIdx', 1 );
            end
        end
        ifProp( iKoopmanRecComponent ) = true;
    end
    nRecKoop = size( propVal{ iKoopmanRecComponent }, 3 );

    % Setup reconstructed component tags, directories, and filenames
    for iRec = 1 : nRecKoop
        for iR = 1 : nR
            for iC = 1 : nCT
                propVal{ iKoopmanRecComponent }( iC, iR, iRec ) = ...
                    setDimension( ...
                        propVal{ iKoopmanRecComponent }( iC, iR, iRec ), ...
                        getDimension( ...
                            parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
                propVal{ iKoopmanRecComponent }( iC, iR, iRec ) = ...
                    setPartition( ...
                        propVal{ iKoopmanRecComponent }( iC, iR, iRec ), ...
                        recPartition( iR ) );
                propVal{ iKoopmanRecComponent }( iC, iR, iRec ) = ...
                    setDefaultTag( ...
                        propVal{ iKoopmanRecComponent }( iC, iR, iRec ) );
                propVal{ iKoopmanRecComponent }( iC, iR, iRec ) = ...
                    setComponentTag( ...
                        propVal{ iKoopmanRecComponent }( iC, iR, iRec ), ...
                        getComponentTag( ...
                            parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
                propVal{ iKoopmanRecComponent }( iC, iR, iRec ) = ...
                    setRealizationTag( ...
                        propVal{ iKoopmanRecComponent }( iC, iR, iRec ), ...
                        getRealizationTag( ...
                            parentConstrArgs{ iTrgComponent }( iC, iR ) ) );    
                propVal{ iKoopmanRecComponent }( iC, iR, iRec ) = ...
                    setDefaultSubpath( ...
                        propVal{ iKoopmanRecComponent }( iC, iR, iRec ) );
                propVal{ iKoopmanRecComponent }( iC, iR, iRec ) = ...
                    setDefaultFile( ...
                        propVal{ iKoopmanRecComponent }( iC, iR, iRec ) );

                pth = strjoin_e( getTag( ...
                    parentConstrArgs{ iTrgEmbComponent }( iC, iR ) ), ...
                    '_' );   
                pth = fullfile( modelPathK, pth, ...
                    getBasisFunctionTag( ...
                        propVal{ iKoopmanRecComponent }( iC, iR, iRec ) ) );
                propVal{ iKoopmanRecComponent }( iC, iR, iRec ) = ...
                    setPath( ...
                        propVal{ iKoopmanRecComponent }( iC, iR, iRec ), pth );
            end
        end
    end
    mkdir( propVal{ iKoopmanRecComponent } )
end



%% LINEAR MAPS
% Setup collective tags for target data
pth = concatenateTags( parentConstrArgs{ iTrgEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetComponentName' )
        if ~isSet
            pth{ 1 } = varargin{ i + 1};
            break
        else
            error( 'Target component name has been already specified' )
        end      
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetRealizationName' )
        if ~isSet
            pth{ 2 } = varargin{ i + 1 };
            break
        else
            error( 'Target realization name has been already specified' )
        end
    end  
end
pth = strjoin_e( pth, '_' );

% Parse linear map templates 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'linearMap' )
        iLinMap = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'linearMapTemplate' )
        if ~isempty( propVal{ iLinMap } )
            error( 'A linear map template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaLinearMap' ) ...
          && isvector( varargin{ i + 1 } )
            propVal{ iLinMap } = varargin{ i + 1 };
        else
            error( 'The linear map template must be specified as a vector of nlsaLinearMap_basis objects' )
        end
        ifProp( iLinMap ) = true;
    end
end
if isempty( propVal{ iLinMap } )
    propVal{ iLinMap } = nlsaLinearMap_gl( 'basisFunctionIdx', 1 : nPhi );
    ifProp( iLinMap ) = true;
end
nA = numel( propVal{ iLinMap } );


for iA = 1 : nA
    propVal{ iLinMap }( iA ) = setPartition( propVal{ iLinMap }( iA ), ...
      partition );
    propVal{ iLinMap }( iA ) = setDimension( ...
        propVal{ iLinMap }( iA ), ...
        getEmbeddingSpaceDimension( parentConstrArgs{ iTrgEmbComponent }( iC, 1 ) ) );
    if ~isCompatible( propVal{ iLinMap }( iA ), propVal{ iPrjComponent } )
        error( 'Incompatible linear map and projected components' )
    end
    modelPathA = fullfile( modelPathL, pth, getDefaultTag( propVal{ iLinMap }( iA ) ) );
    propVal{ iLinMap }( iA ) = setPath( propVal{ iLinMap }( iA ), modelPathA );
    propVal{ iLinMap }( iA ) = setDefaultSubpath( propVal{ iLinMap }( iA ) );
    propVal{ iLinMap }( iA ) = setDefaultFile( propVal{ iLinMap }( iA ) );
end
if ~isCompatible( propVal{ iLinMap } )
    error( 'Incompatible linear maps' )
end
mkdir( propVal{ iLinMap } )

%% SVD RECONSTRUCTED COMPONENTS
% Parse reconstruction templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'svdRecComponent' )
        iSvdRecComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'svdReconstructionTemplate' )
        if ~isempty( propVal{ iSvdRecComponent } )
            error( 'An SVD reconstruction template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
          && isscalar( varargin{ i + 1 } ) 
            propVal{ iSvdRecComponent } = repmat( varargin{ i + 1 }, ...
                                                  [ nCT nR nA ] );
        elseif isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
          && isvector( varargin{ i + 1 } ) ...
          && numel( varargin{ i + 1 } ) == nA
            propVal{ iSvdRecComponent } = repmat( varargin{ i + 1 }, ....
                                                  [ nCT nR 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
          && ismatrix( varargin{ i + 1 } ) ...
          && all( size( varargin{ i + 1 } ) == [ nR nA ] )
            propVal{ iSvdRecComponent } = repmat( varargin{ i + 1 }, ...
                                                  [ nCT 1 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
          && numel( size( varargin{ i + 1 } ) ) == 3 ...
          && all( size( varargin{ i + 1 } ) == [ nCT nR nA ] )
            propVal{ iSvdRecComponent } = varargin{ i + 1 };      
        else
            error( 'The SVD reconstruction template must be specified as a scalar nlsaComponent_rec_phi object, a vector of nlsaComponent_rec_phi objects of size equal to the number of linear maps, or a matirx of nlsaComponent_rec_phi objects of size equal to the number of realizations times the number of linear maps.' )
        end
        ifProp( iSvdRecComponent ) = true;
    end
end
if isempty( propVal{ iSvdRecComponent } )
    for iA = nA : -1 : 1
        for iR = nR : -1 : 1
            for iC = nCT : -1 : 1
                propVal{ iSvdRecComponent }( iC, iR, iA ) = ...
                    nlsaComponent_rec_phi( 'basisFunctionIdx', 1 );
            end
        end
    end
    ifProp( iSvdRecComponent ) = true;
end
for iA = 1 : nA
    for iR = 1 : nR
        for iC = 1 : nCT
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setPartition( ...
               propVal{ iSvdRecComponent }( iC, iR, iA ), recPartition( iR ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setDimension( ...
               propVal{ iSvdRecComponent }( iC, iR, iA ), ...
               getDimension( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setDefaultTag( ...
               propVal{ iSvdRecComponent }( iC, iR, iA ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setComponentTag( ...
                propVal{ iSvdRecComponent }( iC, iR, iA ), ...
                getComponentTag( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setRealizationTag( ...
                propVal{ iSvdRecComponent }( iC, iR, iA ), ...
                getRealizationTag( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );    
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setDefaultSubpath( ...
                propVal{ iSvdRecComponent }( iC, iR, iA ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = setDefaultFile( ...
                propVal{ iSvdRecComponent }( iC, iR, iA ) );

            pth = strjoin_e( ...
                getTag( parentConstrArgs{ iTrgEmbComponent }( iC, iR ) ), '_' );
            pth = fullfile( ... 
                getPath( propVal{ iLinMap }( iA ) ), ...
                pth, ...
                getBasisFunctionTag( propVal{ iSvdRecComponent }( iC, iR, iA ) ) );
            propVal{ iSvdRecComponent }( iC, iR, iA ) = ...
                setPath( propVal{ iSvdRecComponent }( iC, iR, iA ), pth );
        end
    end
end
mkdir( propVal{ iSvdRecComponent } )

%% COLLECT ARGUMENTS
constrArgs                = cell( 1, 2 * nnz( ifProp ) );
constrArgs( 1 : 2 : end ) = propName( ifProp );
constrArgs( 2 : 2 : end ) = propVal( ifProp );
constrArgs = [ constrArgs parentConstrArgs ];
