classdef nlsaModel_ssa < nlsaModel_base

%NLSAMODEL_SSA   Class definition and constructor of SSA model
%
%   nlsaModel_ssa implements Singular Spectrum Analysis as described in 
%   references [1,2] below.
%
%   Following the Takens delay embedding implemented by the parent class 
%   nlsaModel_base this class implements methods to carry out the main steps
%   of NLSA algorithms, namely:
% 
%   (i)   Calculation of the covariance matrix for the source data, in either
%         temporal or spatial form, and calculation of the associated 
%         eigenfunctions.
%   (ii)  Projection of the target data onto the eigenfunctions of the 
%         temporal covariance matrix. 
%   (iii) Singular value decomposition (SVD) of the projected data and 
%         calculation of the associated spatial and temporal patterns. 
%   (iv)  Reconstruction of the eigenfunction projected data and the SVD data.
%   The class constructor arguments are passed as property name-property value
%   pairs using the syntax
%
%   model = nlsaModel( propName1, propVal1, propName2, propVal2, ... ).
%
%   In addition to the of the nlsaModel_base parent class, the following 
%   properties can be specified. 
%
%  'prjComponent': An nlsaProjection object storing the projections of the 
%      target data onto the temporal covariance  eigenfunctions. This property
%      must be specified as a vector of size [ nCT 1 ], where nCT is the 
%      number of target components in the model. nlsaProjection objects have a
%      property nDE specifying the dimension of the projected data in Takens
%      delay embedding space. This property must be consistent with the delay
%      space dimensions of the target data. That is prjComponent( iCT ).nDE
%      must be equal to the value returned by the function 
%      getEmbeddingSpaceDimension( trgEmbComponent( iCT, iR ) ) for all 
%      realizations iR.  
%              
%  'linearMap': An nlsaLinearMap object implementing the SVD of the projected
%      target data and the computation of the associated temporal patterns.
%      Currently, these objects are implemented for the 'gl' storage fornmat,
%      so this property must be set to an nlsaLinearMap_gl object.
%      nlsaLinearMap objects have a property idxPhi which is a vector of
%      integers specifying the eigenfunctions used in SVD. The elemets of
%      idxPhi must be distinct positive integers less than the propery nEig of
%      the model's covariance operator. nlsaLinearMap objects also have a
%      partition propery which must be identical to the partition property of
%      the model's covariance operator. 
%
%      The property linMap can be either a scalar or a vector. In the 
%      latter case, the eigenfunctions of the linear maps must be nested, i.e.,
%      linMap( iA ).idxPhi( 1 : end - 1 ) = linMap( iA - 1 ).idxPhi( : ). This
%      can speed up file I/O when performing SVD of multiple linear maps. 
%      
%   'recComponent': An [ nCT nR ]-sized array of nlsaComponent_rec_phi objects 
%      implementing the reconstruction of the projected data onto the
%      covariance eigenfunctions. The nlsaComponent_rec_phi class is a child 
%      of the nlsaComponent class. In particular, it has a dimension property
%      nD which must be compatible with the dimension of the target data, i.e., 
%      recComponent( iCT, iR ).nD must be equal to trgComponent( iCT,iR ).nD.   
%      nlsaComponent_rec objects also have a partition property which must be
%      set to an nlsaPartition object. The number of samples in 
%      recComponent( iCT, iR ).partition must not exceed the number of samples
%      in the delay embedded data, trgEmbComponent( iCT, iR ).partition, plus
%      nE( iCT ) - 1, where nE( iCT ) is the number of delays for target 
%      component iCT. 


%   References
%   [1] N. Aubry et al. (1992), "Spatiotemporal analysis of complex signals: 
%       Theory and applications," J. Stat. Phys., 64(3/4), 683
%   [2] M. Ghil et al., "Advanced spectral methods for climatic time series", 
%       Rev.\ Geophys., 40(1), 2002
%
%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2016/06/02

    properties
        covOp           = nlsaCovarianceOperator_gl();
        prjComponent    = nlsaProjectedComponent();
        recComponent    = nlsaComponent_rec_phi();
        linMap          = nlsaLinearMap_gl();
        svdRecComponent = nlsaComponent_rec_phi();
    end

    methods
        
        %% CLASS CONSTRUCTOR
        function obj = nlsaModel_ssa( varargin )

            msgId = 'nlsa:nlsaModel_ssa:';

            % Check if constructor is called in "template" mode, and parse
            % templates if needed
            if ifTemplate( 'nlsaModel_ssa', varargin{ : } )
                varargin = nlsaModel_ssa.parseTemplates( varargin{ : } );
            end
            
            nargin   = numel( varargin );
            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iCovOp           = [];
            iLinearMap       = [];
            iPrjComponent    = [];
            iRecComponent    = [];
            iSvdRecComponent = [];

            for i = 1 : 2 : nargin
                switch varargin{ i } 
                    case 'covarianceOperator'
                        iCovOp = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'linearMap'
                        iLinearMap = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'prjComponent'
                        iPrjComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'recComponent'
                        iRecComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'svdRecComponent'
                        iSvdRecComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end


            obj = obj@nlsaModel_base( varargin{ ifParentArg } );

            partition    = getEmbPartition( obj ); 
            nSETot       = getNEmbSample( obj ); % total no. of samples in embedding space
            trgEmbComponent = getTrgEmbComponent( obj ); 
            embComponent = getEmbComponent( obj );

            % Covariance operator
            if ~isempty( iCovOp )
                if ~isa( varargin{ iCovOp }, 'nlsaCovarianceOperator' ) ...
                       && isscalar( varargin{ iCovOp } )
                       error( [ msgId 'invalidDiffOp' ], ...
                              'The covarianceOperator property must be specified as a scalar nlsaCovarianceOperator object' )
                end
                obj.covOp = varargin{ iCovOp };
            else
                nDE = getEmbeddingSpaceDimension( embComponent( :, 1 ) );
                for iC = 2 : numel( nDE );
                    nDE( iC ) = nDE( iC ) - 1 + nDE( iC - 1 );
                end
                partitionD = mergePartitions( partitionD );  
                obj.covOp = nlsaCovarianceOperator_gl( 'partition', partition, ...
                                                    'nEigenfunction', min( 10, nSETot ) ); 

                                                    'spatialPartition', nlsaPartition( 'idx', nDE );
            end

            % Projected component
            if ~isempty( iPrjComponent )
                if ~isa( varargin{ iPrjComponent }, 'nlsaProjectedComponent' )
                    error( [ msgId 'invalidPrj' ], ...
                        'Projected data must be specified as nlsaProjectedComponent objects' )
                end
                if ~isCompatible( varargin{ iPrjComponent }, trgEmbComponent )
                    error( 'Incompatible projected and target embedded components' )
                end
                if ~isCompatible( varargin{ iPrjComponent }, obj.covOp )
                    error( 'Incompatible projected components and covariance operator' )
                end
                obj.prjComponent = varargin{ iPrjComponent };
            else
                nCT = size( trgEmbComponent, 1 );
                nDE = getEmbeddingSpaceDimension( trgEmbComponent( :, 1 ) );
                nL  = getNEigenfunction( obj.covOp );
                for iC = nCT : -1 : 1
                    obj.prjComponent( iC ) = nlsaProjectedComponent( ...
                        'embeddingSpaceDimension', nDE( iC ), ...
                        'partition', partition, ...
                        'nBasisFunction', nL );
                end
                obj.prjComponent = obj.prjComponent';
            end

            % Reconsructed component
            if ~isempty( iRecComponent )
                if ~isa( varargin{ iRecComponent }, 'nlsaComponent_rec_phi' )
                    error( [ msgId 'invalidRecOmponent' ], ...
                        'Reconstructed component must be specified as an array of of nlsaComponent_rec_phi objects.' )        
                end
                if ~isCompatible( trgEmbComponent, varargin{ iRecComponent } )
                    error( 'Incompatible reconstructed components' )
                end
                obj.recComponent = varargin{ iRecComponent };
            end

            % Linear map
            if ~isempty( iLinearMap )
                if ~isa( varargin{ iLinearMap }, 'nlsaLinearMap' ) ...
                  || ~isvector( varargin{ iLinearMap } )
                    error( [ msgId 'invalidLinMap' ], ...
                        'Linear map must be specified as a vector of nlsaLinearMap objects.' )
                end
                if ~isCompatible( varargin{ iLinearMap }, obj.prjComponent )
                    error( 'Incompatible linear map and projected components' )
                end
                obj.linMap = varargin{ iLinearMap };
            else
                nDE = getTrgEmbeddingSpaceDimension( obj );
                for iC = 2 : numel( nDE );
                    nDE( iC ) = nDE( iC ) - 1 + nDE( iC - 1 );
                end

                obj.linMap = nlsaLinearMap( ...
                  'basisFunctionIdx', 1 : getNEigenfunction( obj.covOp ), ...
                  'spatialPartition', nlsaPartition( 'idx', nDE ) );
            end

            % Reconsructed component from SVD
            if ~isempty( iSvdRecComponent )
                if ~isa( varargin{ iSvdRecComponent }, 'nlsaComponent_rec_phi' )
                    error( [ msgId 'invalidSvdRecOmponent' ], ...
                        'Reconstructed component must be specified as an array of of nlsaComponent_rec_phi objects.' )        
                end
                for iA = 1 : numel( obj.linMap )
                    if ~isCompatible( trgEmbComponent, ...
                          squeeze( varargin{ iSvdRecComponent }( :, :, iA ) ) )
                        error( 'Incompatible reconstructed components' )
                    end
                end
                obj.svdRecComponent = varargin{ iSvdRecComponent };
            end
        end
    end

    methods( Static )
 
        %% LISTCONSTRUCTORPROPERTIES List property names for class constructor
        function pNames = listConstructorProperties
            pNames = nlsaModel_base.listConstructorProperties;
            pNames = [ pNames ...
                       { 'covarianceOperator' ...
                         'prjComponent' ...
                         'recComponent' ...
                         'linearMap' ...
                         'svdRecComponent' } ];
        end

        %% LISTPARSERPROPERTIES  List property names for class constructor parser
        function pNames = listParserProperties
            pNames = nlsaModel_base.listParserProperties;
            pNames = [ pNames ...
                       { 'sourceComponentName' ...
                         'sourceRealizationName' ...
                         'covarianceOperatorTemplate' ...
                         'projectionTemplate' ...
                         'reconstructionTemplate' ...
                         'reconstructionPartition' ...
                         'targetComponentName' ...
                         'targetRealizationName' ...
                         'linearMapTemplate' ...
                         'svdReconstructionTemplate' } ];
        end

        %% PARSETEMPLATES  Template parser
        propNameVal = parseTemplates( varargin );        
    end
end

