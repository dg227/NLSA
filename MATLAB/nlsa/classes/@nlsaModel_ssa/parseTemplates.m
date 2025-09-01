function constrArgs = parseTemplates( varargin ) 
%PARSETEMPLATES Construct property name-value pairs for the nlsaModel_ssa class
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
%   'covarianceOperatorTemplate': An nlsaCovarianceOperator object specifying
%      the method (e.g., global or batch) used to compute the temporal and
%      spatial covariance matrix for the source data.
%
%   'projectionTemplate': An array of nlsaProjection objects specifying the
%      projections of the target data in delay-embedding space onto the
%      covariance eigenfunctions. 'projectionTemplate' must be a vector of size 
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
%   'projectionComponentName': Similar to 'targetComponentName', but used to
%      compress the component-dependent part of the projected data directory.
%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2016/06/03


%% CONSTRUCTOR PROPERTY LIST
propName = nlsaModel_ssa.listConstructorProperties; 
propVal  = cell( size( propName ) );
ifProp   = false( size( propName ) );
idxProp  = zeros( size( propName ) );
nProp    = numel( idxProp );
%% SUPERCLASS CONSTRUCTOR ARGUMENTS
parentConstrArgs = nlsaModel_base.parseTemplates( varargin{ : } );
for iProp = 1 : 2 : numel( parentConstrArgs )
    switch parentConstrArgs{ iProp }
        case 'embComponent'
            iEmbComponent = iProp + 1;
        case 'trgComponent'
            iTrgComponent = iProp + 1;
        case 'trgEmbComponent'
            iTrgEmbComponent = iProp + 1;
    end
end
partition = getPartition( parentConstrArgs{ iEmbComponent }( 1, : ) );
nCT = size( parentConstrArgs{ iTrgEmbComponent }, 1 );
nC  = size( parentConstrArgs{ iEmbComponent }, 1 );
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

%% COVARIANCE OPERATOR
% Parse covariance operator template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'covarianceOperator' )
        iCovOp = iProp;
        break
    end  
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'covarianceOperatorTemplate' )
        if ~isempty( propVal{ iCovOp } )
            error( 'A diffusion operator template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaCovarianceOperator' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iCovOp } = varargin{ i + 1 };
        else
            error( 'The diffusion operator template must be specified as a scalar nlsaCovarianceOperator object' )
        end
        ifProp( iCovOp ) = true;
    end
end
if isempty( propVal{ iCovOp } )
    propVal{ iCovOp } = nlsaCovarianceOperator_gl();
    ifProp( iCovOp ) = true;
end
propVal{ iCovOp } = setPartition( propVal{ iCovOp }, partition );
nDECum = getEmbeddingSpaceDimension( parentConstrArgs{ iEmbComponent }( :, 1 ) );
for iC = 2 : nC
    nDECum( iC ) = nDECum( iC ) - 1 + nDECum( iC - 1 );
end
propVal{ iCovOp } = setSpatialPartition( ...
                      propVal{ iCovOp }, ...
                      nlsaPartition( 'idx', nDECum ) );                   
nPhi   = getNEigenfunction( propVal{ iCovOp } );
tag = getTag( propVal{ iCovOp } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iCovOp } = setTag( propVal{ iCovOp }, ...
        [ tag getDefaultTag( propVal{ iCovOp } ) ] ); 


% Determine covariance-specific directories
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
modelPathC = fullfile( modelPath, 'processed_data', pth, ...
                       getTag( propVal{ iCovOp } )  );
propVal{ iCovOp } = setPath( propVal{ iCovOp }, modelPathC );
propVal{ iCovOp } = setDefaultSubpath( propVal{ iCovOp } ); 
propVal{ iCovOp } = setDefaultFile( propVal{ iCovOp } );
mkdir( propVal{ iCovOp } )



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
for iC = 1 : nCT
    % Setup collective tags for projected data
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'projectionComponentName' )
            if ~isSet
                pth = varargin{ i + 1 };
                break
            else
                error( 'Projection component name has been already specified' )
            end      
        end  
    end
    if ~isSet
        pth = concatenateTags( parentConstrArgs{ iTrgEmbComponent }( iC, : ) );
        pth = strjoin_e( pth, '_' );
    end

    propVal{ iPrjComponent }( iC ) = setPartition( propVal{ iPrjComponent }( iC ), partition );
    propVal{ iPrjComponent }( iC ) = setPath( propVal{ iPrjComponent }( iC ), ...
    fullfile( modelPathC, pth ) );
    propVal{ iPrjComponent }( iC ) = setEmbeddingSpaceDimension( ...
       propVal{ iPrjComponent }( iC ), ...
       getEmbeddingSpaceDimension( parentConstrArgs{ iTrgEmbComponent }( iC, 1 )) );
    propVal{ iPrjComponent }( iC ) = setDefaultSubpath( propVal{ iPrjComponent }( iC ) );
    propVal{ iPrjComponent }( iC ) = setDefaultFile( propVal{ iPrjComponent }( iC ) );
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
            && isscalar( varargin{ i + 1 } ) 
            propVal{ iRecComponent } = repmat( varargin{ i + 1 }, [ nCT nR ] );
        elseif isa( varargin{ i + 1 }, 'nlsaComponent_rec_phi' ) ...
           && isvector( varargin{ i + 1 } ) ...
           && numel( varargin{ i + 1 } ) == nR
            propVal{ iRecComponent } = varargin{ i + 1 };
            if isrow( propVal{ iRecComponent } )
                propVal{ iRecComponent } = propVal{ iRecComponent }';
            end
            propVal{ iRecComponent } = repmat( propVal{ iRecComponent }, ...
                                               [ nCT 1 ] );
        else
            error( 'The reconstruction template must be specified as a scalar nlsaComponent_rec_phi object, or vector of nlsaComponent_rec_phi objects of size equal to the number of realizations' )
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
    for iC = 1 : nCT
        propVal{ iRecComponent }( iC, iR ) = setPartition( propVal{ iRecComponent }( iC, iR ), recPartition( iR ) );
    end
end

% Setup reconstructed component tags, directories, and filenames
for iR = 1 : nR
    for iC = 1 : nCT
        propVal{ iRecComponent }( iC, iR ) = setDimension( ...
           propVal{ iRecComponent }( iC, iR ), ...
           getDimension( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
        propVal{ iRecComponent }( iC, iR ) = setDefaultTag( ...
            propVal{ iRecComponent }( iC, iR ) );
        propVal{ iRecComponent }( iC, iR ) = setComponentTag( ...
            propVal{ iRecComponent }( iC, iR ), ...
            getComponentTag( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
        propVal{ iRecComponent }( iC, iR ) = setRealizationTag( ...
            propVal{ iRecComponent }( iC, iR ), ...
            getRealizationTag( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );    
        propVal{ iRecComponent }( iC, iR ) = setDefaultSubpath( ...
            propVal{ iRecComponent }( iC, iR ) );
        propVal{ iRecComponent }( iC, iR ) = setDefaultFile( ...
            propVal{ iRecComponent }( iC, iR ) );

        pth = strjoin_e( ...
            getTag( parentConstrArgs{ iTrgEmbComponent }( iC, iR ) ), '_' );
        pth = fullfile( modelPathC, ...
            pth, ...
            getBasisFunctionTag( propVal{ iRecComponent }( iC, iR ) ) );
        propVal{ iRecComponent }( iC, iR ) = ...
            setPath( propVal{ iRecComponent }( iC, iR ), pth );
    end
end
mkdir( propVal{ iRecComponent } )

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
    modelPathA = fullfile( modelPathC, pth, getDefaultTag( propVal{ iLinMap }( iA ) ) );
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
