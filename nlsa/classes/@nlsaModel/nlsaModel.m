classdef nlsaModel < nlsaModel_base
%NLSAMODEL   Class definition and constructor of NLSA model
%
%   nlsaModel implements a generic version of Nonlinear Laplacian Spectral
%   Analysis (NLSA) algorithms, as described in references [1,2] below. 
% 
%   Following the Takens delay embedding implemented by the parent class 
%   nlsaModel_base this class implements methods to carry out the main steps
%   of NLSA algorithms, namely:
% 
%   (i)   Calculation of pairwise distances for the source data, truncated to
%         a specified number of nearest neighbors.
%   (ii)  Symmetrization of the pairwise distances.
%   (iii) Construction of a kernel diffusion operator from the distance data
%         and computation of its eigenfunctions. 
%   (iv)  Projection of the target data onto the diffusion eigenfunctions.
%   (v)   Singular value decomposition (SVD) of the projected data and 
%         calculation of the associated spatial and temporal patterns. 
%   (vi)  Reconstruction of the eigenfunction projected data and the SVD data.
%
%   The class constructor arguments are passed as property name-property value
%   pairs using the syntax
%
%   model = nlsaModel( propName1, propVal1, propName2, propVal2, ... ).
%
%   In addition to the of the nlsaModel_base parent class, the following 
%   properties can be specified.
%   
%   'pairwiseDistance': An nlsaPairwiseDistance object specifying
%      the method and parameters (e.g., number of nearest neighbors) used 
%      to compute the pairwise distances for the delay-embedded data. 
%      'pairwiseDistance' operates on the data specified in property 
%      'embComponent' of the model. nlsaPairwiseDistance objects have a 
%      property partition which must be set to a vector of nlsaPartition
%      objects of size [ 1 nR ], where nR is the number of realizations. These 
%      partitions must conform with the corresponding partitions in the 
%      embedded source data in the sense that pairwiseDistance.partition( iR )
%      must be identical to embComponent( iC, iR ).partition for all 
%      components iC. Pairwise distances are computed in a block fashion for
%      the elements of the partition. 
%
%      In addition, nlsaPairwiseDistance objects have a property dFunc which
%      is set to an nlsaLocalDistanceFunction object specifying the function
%      used to compute pairwise distances (see below for additional details).
%      nlsaPairwiseDistance objects also have a proprety nN specifying the 
%      number of nearest neighbor distances retained for each data point. In
%      general, the memory requirements for the pairwise distance calculation 
%      for a partition batch with nSB data samples scale as nN * nSB. If 
%      'pairwiseDistance' is not specified, the local distance function is 
%      set to the L2 norm distance, and the number of nearest neighnors nN to
%      1/10 of the number of delay-embeded samples. 
%
%   'symmetricDistance': An nlsaSymmetricDistance object 
%      implementing the pairwise distance symmetrization. nlsaSymmetricDistance
%      objects have a partition property which must be identical to the 
%      corresponding property of the model's pairwiseDistance property. Two 
%      classes, nlsaSymmetricDistance_gl and nlsaSymmetricDistance_batch are
%      derived from nlsaSymmetricDistance implementing different storage 
%      formats for the symmetrized distance data. In the case of 
%      nlsaSymmetricDistance_gl objects, the distance data are stored in 
%      global arrays. In the case of nlsaSymmetricDistance_batch, the distance
%      data are stored batchwise for the elements of the partition. In general,
%      the 'gl' storage format is faster than 'batch' but requires sufficient
%      memory to hold the global distance data.
%
%      Objects of class nlsaSymmetricDistance have a property nN which 
%      specifies the number of nearest neighbors per data point retained from
%      the original pairwise distances in the distance symmetrization process.
%      nN cannot exceed the value of the corresponding property of the 
%      nlsaPairwiseDistance object used in the object. 
%
%      Objects of class nlsaSymmetricDistance_batch also have a property
%      nNMax which specifies the maximum number of nearest neighbors that can 
%      be stored after distance symmetrization. The memory required for 
%      symmetrization with this class is the equivalent of nSBmax * nNMax 
%      double precision numbers, where nSB max is the maximum number of 
%      samples in the partition. 
% 
%      If 'symmetricDistance' is not specified, it is set to an 
%      nlsaSymmetricDistance_gl object with nN equal to the corresponding 
%      property of the pairwise distance used in the model. 
%
%   'diffOp': An nlsaDiffusionOperator object implementing the kernel 
%      diffusion operator for the model. Two derived classes,
%      nlsaDiffusionOperator_gl and nlsaDiffusionOperator_batch are provided
%      for global and batch storage format, respectively. An additional class, 
%      nlsaDiffusionOperator_gl_mb is derived from nlsaDiffusionOperator_gl
%      implementing the automatic bandwidth selection procedure in [3] using
%      multiple kernel bandwidths ('mb'). 
% 
%      All nlsaDiffusionOperator objects have a property partition which
%      must be identical to the corresponding property of the 
%      nlsaSymmetric distance object in the model. In addition, 
%      nlsaDiffusionOperator objects have a property nEig specifying the
%      number of eigenvectors to be computed. 
%       
%      nlsaDiffusionOperator_gl objects have a kernel bandwidth property
%      epsilon and a normalization parameter alpha as described in the
%      diffusion maps algorithm [4]. nlsaDiffusionOperator_gl_mb have two
%      additional bandwidth related properties, epsilonB and epsilonELim,
%      specifying the base and exponent limits in the bandwidth selection 
%      process.  
%
%      nlsaDiffusionOperator_batch objects have a property nN specifying 
%      the number of graph edges ("nearest neighbors") per data sample. This
%      property must be set equal to the property nNMax of the 
%      nlsaSymmetricDistance_batch object used in the model. The class
%      nlsaDiffusionOperator_batch provides distributed eigenvector calcultion
%      using Matlab's SPMD features. See the class method 
%      computeDiffusionEigenfunctions_spmd for additional details. 
% 
%  'prjComponent': An nlsaProjection object storing the projections of the 
%      target data onto the diffusion eigenfunctions. This property must be 
%      specified as a vector of size [ nCT 1 ], where nCT is the number of
%      target components in the model. nlsaProjection objects have a property
%      nDE specifying the dimension of the projected data in Takens delay
%      embedding space. This property must be consistent with the delay
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
%      the model's diffusion operator. nlsaLinearMap objects also have a
%      partition propery which must be identical to the partition property of
%      the model's diffusion operator. 
%
%      The property linMap can be either a scalar or a vector. In the 
%      latter case, the eigenfunctions of the linear maps must be nested, i.e.,
%      linMap( iA ).idxPhi( 1 : end - 1 ) = linMap( iA - 1 ).idxPhi( : ). This
%      can speed up file I/O when performing SVD of multiple linear maps. 
%      
%   'recComponent': An [ nCT nR ]-sized array of nlsaComponent_rec_phi objects 
%      implementing the reconstruction of the projected data onto the
%      diffusion eigenfunctions. The nlsaComponent_rec_phi class is a child 
%      of the nlsaComponent class. In particular, it has a dimension property
%      nD which must be compatible with the dimension of the target data, i.e., 
%      recComponent( iCT, iR ).nD must be equal to trgComponent( iCT,iR ).nD.   
%      nlsaComponent_rec objects also have a partition property which must be
%      set to an nlsaPartition object. The number of samples in 
%      recComponent( iCT, iR ).partition must not exceed the number of samples
%      in the delay embedded data, trgEmbComponent( iCT, iR ).partition, plus
%      nE( iCT ) - 1, where nE( iCT ) is the number of delays for target 
%      component iCT. 
%
%      nlsaComponent_rec_phi objects have a property idxPhi specifying the 
%      diffusion eigenfunctions used for reconstruction. recComponent.idxPhi
%      must be a vector of distinct positive integers less than the property
%      nEig of the model's diffusion operator.
%
%   'svdRecComponent': Same as 'recComponent', but in this case reconstruction
%      is performed using the spatial and temporal patterns from the SVD of the
%      model's linar maps. svdRecComponent is an array of nlsaComponent_rec 
%      objects of size [ nCT nR nA ] where nA is the number of elements of 
%      linMap. The property svdRecComponent( iCT, iR, iA ).idxPhi specifies the
%      indices of the SVD modes of linMap( iA ) used for reconstruction. This 
%      property must be a vector of distinct integers less than or equal to 
%      the number of elements in linMap( iA ).idxPhi. 
%       
%   Alternatively, the constructor can be called in "template" mode, where 
%   instead of the fully defined objects listed above the arguments supplied
%   by the user only have a set of essential properties defined, and the 
%   remaining properties are filled in automatically. See the class method 
%   parseTemplates for more detais. 
%
%   Below is a summary of selected methods implemented by this class. These
%   methods can be executed in the sequence listed below. The results of each
%   step are written on disk.  
%
%   - computePairwiseDistances: Computes pairwise distances for the source
%     delay-embedded data. This mehod supports rudimentary parallelization
%     by splitting the calculation into blocks which can be executed in 
%     parallel. 
%
%   - symmetrizeDistances: Symmetrizes the pairwise distances. This method
%     supports basic parallelization features when using the 
%     nlsaSymmetricDistance_batch class.
%
%   - computeKernelNormalization, computeKernelDegree: Computes the kernel
%     normalization and kernel degree used in diffusion maps. These steps 
%     should be only executed when using diffusion operators of class 
%     nlsaDiffusionOperator_batch.
%
%   - computeKernelDobuleSum: Computes the kernel sum for trial bandwidth
%     values. This step is needed for automatic bandiwdth selection and should
%     only be executed with diffusion operators of class 
%     nlsaDiffusionOperator_gl_mb. 
%
%   - computeDiffusionOperator: Computes the diffusion operator from the 
%     symmetric distance data. This step is mandatory when the diffusion 
%     operator is of class nlsaDiffusionOperator_batch. In the case of 
%     diffusion operators of class nlsaDiffusionOperator_gl this step is 
%     optional and can be omitted if it is not desired to save the matrix
%     elements of the diffusion operator on disk. 
%
%   - computeDiffusionEigenfunctions: Solves the eigenvalue problem for the 
%     diffusion operator. 
%
%   - computeDiffusionEigenfunctions_spmd: An alternative to 
%     computeDiffusionEigenfunctions that uses Matlab's SPMD features. Only
%     supported for diffusion operators of class nlsaDiffusionOperator_batch. 
% 
%   - computeProjection: Computes the projection of the target data in 
%     trgEmbComponent on the diffusion eigenfunctions.
%
%   - computeReconstruction: Reconstucts the eigenfunction projected target
%     data. 
% 
%   The following steps are only needed if SVD of the eigenfunction projected
%   data [1,2] is required. 
%
%   - computeSVD: Computes the SVD of the eigenfunction projected target data.
%
%   - computeSVDTemporalPatterns: Computes the temporal patterns from the 
%     right singular vectors. 
%
%   - computeSVDReconstruction: Performs reconstruction of the SVD modes.
%
%   The following methods can be used to access the results of the calculation:
% 
%   - getDiffusionEigenfunctions: Retrieves the diffusion eigenfunctions.
%
%   - getProjectedData: Retrieves the eigenfunction projected target data. 
%
%   - getReconstructedData: Retrieves the reconstructed data from the 
%     diffusion eigenfunctions.     
%
%   - getSVDTemporalPatterns: Retrieves the SVD temporal patterns.
%
%   - getSVDReconstructedData: Retrieves the reconstructed data from the 
%     SVD modes.
%
% 
%   References
%   [1] D. Giannakis and A. J. Majda (2012), "Nonlinear Laplacian spectral 
%       analysis for time series with intermittency and low-frequency 
%       variability", Proc. Natl. Acad. Sci., 109(7), 2222, 
%       doi:10.1073/pnas.1118984109 
%   [2] D. Giannakis and A. J. Majda (2012), "Nonlinear Laplacian spectral 
%       analysis: Capturing intermittent and low-frequency spatiotemporal 
%       patterns in high-dimensional data", Stat. Anal. and Data Min., 
%       doi:10.1002/sam.11171
%   [3] T. Berry and J. Harlim (2015), "Variable bandwidth diffusion kernels",
%       Appl. Comput. Harmon. Anal., doi:10.1016/j.acha.2015.01.001
%   [4] R. R. Coifman and S. Lafon (2006), "Diffusion Maps", Appl. 
%       Comput. Harmon. Anal., 21, 5, doi:10.1016/j.acha.2006.04.006
%
%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2019/11/20

    %% PROPERTIES
    properties
        pDistance       = nlsaPairwiseDistance();
        sDistance       = nlsaSymmetricDistance_gl();
        diffOp          = nlsaDiffusionOperator_gl();
        prjComponent    = nlsaProjectedComponent();
        recComponent    = nlsaComponent_rec_phi();
        linMap          = nlsaLinearMap_gl();
        svdRecComponent = nlsaComponent_rec_phi();
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaModel( varargin )
           
            msgId = 'nlsa:nlsaModel:';

            % Check if constructor is called in "template" mode, and parse
            % templates if needed
            if ifTemplate( 'nlsaModel', varargin{ : } )
                varargin = nlsaModel.parseTemplates( varargin{ : } );
            end
            
            nargin   = numel( varargin );
            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iPDistance       = [];
            iSDistance       = [];
            iDiffOp          = [];
            iLinearMap       = [];
            iPrjComponent    = [];
            iRecComponent    = [];
            iSvdRecComponent = [];

            for i = 1 : 2 : nargin
                switch varargin{ i } 
                    case 'pairwiseDistance'
                        iPDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'symmetricDistance'
                        iSDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'diffusionOperator'
                        iDiffOp = i + 1;
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

            partition       = getEmbPartition( obj ); 
            partitionT      = getEmbPartitionT( obj );
            partitionQ      = getEmbPartitionQ( obj );
            nSETot          = getNEmbSample( obj ); 
            trgEmbComponent = getTrgEmbComponent( obj ); 

            % Pairwise distance
            if ~isempty( iPDistance )
                if ~isa( varargin{ iPDistance }, 'nlsaPairwiseDistance' ) ...
                       && isscalar( varargin{ iPDistance } )
                       error( [ msgId 'invalidPDist' ], ...
                              'The pairwise distance property must be specified as a scalar nlsaPairwiseDistance object' )
                end
                if getNNeighbors( varargin{ iPDistance } ) ...
                   > getNTotalSample( varargin{ iPDistance } )
                    error( 'The number of nearest neighbors cannot exceed the number of samples' )
                end
                obj.pDistance = varargin{ iPDistance };
            else
                obj.pDistance = nlsaPairwiseDistance_gl( ...
                    'partition',        partitionQ, ...
                    'partitionT',       partitionT, ...
                    'nearestNeighbors', round( nSETot / 10 ) ); 
            end

            % Symmetric distance
            if ~isempty( iSDistance )
                if ~isa( varargin{ iSDistance }, 'nlsaSymmetricDistance' ) ...
                       && isscalar( varargin{ iSDistance } )
                       error( [ msgId 'invalidSDist' ], ...
                              'The symmetric distance property must be specified as a scalar nlsaSymmetricDistance object' )
                end
                if getNNeighbors( varargin{ iSDistance } ) ...
                  > getNNeighbors( obj.pDistance )
                    error( 'The number of nearest neighbors in the symmetric distances cannot exceed the number of nearest neighbors in the pairwise distances' )
                end
                obj.sDistance = varargin{ iSDistance };
            else
                obj.sDistance = nlsaSymmetricDistance_gl( ...
                    'partition', partitionQ, ...
                    'nearestNeighbors', getNNeighbors( obj.pDistance ) ); 
            end

            % Diffusion operator
            if ~isempty( iDiffOp )
                if ~isa( varargin{ iDiffOp }, 'nlsaDiffusionOperator' ) ...
                       && isscalar( varargin{ iDiffOp } )
                       error( [ msgId 'invalidDiffOp' ], ...
                              'The diffusionOperator property must be specified as a scalar nlsaDiffusionOperator object' )
                end
                obj.diffOp = varargin{ iDiffOp };
            else
                obj.diffOp = nlsaDiffusionOperator_gl( ...
                    'partition', partition, ...
                    'nEigenfunction', min( 10, nSETot ) ); 
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
                if ~isCompatible( varargin{ iPrjComponent }, obj.diffOp )
                    error( 'Incompatible projected components and diffusion operator' )
                end
                obj.prjComponent = varargin{ iPrjComponent };
            else
                nCT = size( trgEmbComponent, 1 );
                nDE = getEmbeddingSpaceDimension( trgEmbComponent( :, 1 ) );
                nL  = getNEigenfunction( obj.diffOp );
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
                  'basisFunctionIdx', 1 : getNEigenfunction( obj.diffOp ), ...
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
                       { 'pairwiseDistance' ...
                         'symmetricDistance' ...
                         'diffusionOperator' ...
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
                         'pairwiseDistanceTemplate' ...
                         'symmetricDistanceTemplate' ...
                         'diffusionOperatorTemplate' ...
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
