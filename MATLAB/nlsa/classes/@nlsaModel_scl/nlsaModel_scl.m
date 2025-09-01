classdef nlsaModel_scl < nlsaModel_err
%NLSAMODEL_SCL   Class definition and constructor of NLSA model with
%  scaled kernels by dynamical model error  
%
% sclOutPDistance: Scaled pairwise distance for OS data by state/vel error 
% sclOutSDistance: Symmetric distance associated with sclOutPDistance
% sclOutDiffOp:    Diffusion operator on OS data associated with sclOutSDistance
% sclOutPrjComponent: Projected target data onto eigenfunctions of sclOutDiffOp
% sclIsrPDistance: Scaled pairwise distances for in-sample restriction
% sclIsrDiffOp:    Scaled in-sample restriction operator


% Modified 2014/10/13

    %% PROPERTIES
    properties

        sclOutPDistance    = nlsaPairwiseDistance_scl();  % scaled distance
        sclOutSDistance    = nlsaSymmetricDistance_gl();  % symmetric distance
        sclOutDiffOp       = nlsaDiffusionOperator_gl();  % diffusion operator
        sclOutPrjComponent = nlsaProjectedComponent(); 
        sclIsrPDistance    = nlsaPairwiseDistance_scl();
        sclIsrDiffOp       = nlsaDiffusionOperator_ose();
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = nlsaModel_scl( varargin )
            
            msgId = 'nlsa:nlsaModel_scl:';
            
            % Check if constructor is called in "template" mode and parse
            % templates if needed
            if ifTemplate( 'nlsaModel_scl', varargin{ : } )
                varargin = nlsaModel_scl.parseTemplates( varargin{ : } );
                nargin   = numel( varargin );
            end
  
            ifParentArg = true( 1, nargin ); 

            % Parse input arguments 
            iSclOutPDistance    = [];
            iSclOutSDistance    = [];
            iSclOutDiffOp       = [];
            iSclOutPrjComponent = [];
            iSclIsrPDistance    = [];
            iSclIsrDiffOp       = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }                    
                   case 'sclOutPairwiseDistance'
                        iSclOutPDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'sclOutSymmetricDistance'
                        iSclOutSDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'sclOutDiffusionOperator'
                        iSclOutDiffOp = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'sclOutPrjComponent'
                        iSclOutPrjComponent = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'sclIsrPairwiseDistance'
                        iSclIsrPDistance = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'sclIsrDiffusionOperator'
                        iSclIsrDiffOp = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaModel_err( varargin{ ifParentArg } ); 

            nS  = getNEmbSample( obj );
            nSO = getNOutEmbSample( obj );

            % Scaled pairwise distance
            if ~isempty( iSclOutPDistance )
                if ~isa( varargin{ iSclOutPDistance }, 'nlsaPairwiseDistance_scl' ) ...
                       && ~isscalar( varargin{ iSclOutPDistance } )
                       error( [ msgId 'invalidSclPDist' ], ...
                              'The scaled pairwise distance property must be specified as a scalar nlsaPairwiseDistance_scl object' )
                end
                obj.sclOutPDistance = varargin{ iSclOutPDistance };
            else
                obj.sclOutPDistance = nlsaPairwiseDistance( ...
                                    'partition', partitionO, ...
                                    'nearestNeighbors', round( nSO / 10 ) ); 
            end

            % Scaled symmetric distance
            if ~isempty( iSclOutSDistance )
                if ~isa( varargin{ iSclOutSDistance }, 'nlsaSymmetricDistance' ) ...
                       && isscalar( varargin{ iSclOutSDistance } )
                       error( [ msgId 'invalidSDist' ], ...
                              'The scaledsymmetric distance property must be specified as a scalar nlsaSymmetricDistance object' )
                end
                obj.sclOutSDistance = varargin{ iSclOutSDistance };
            else
                obj.sclOutSDistance = nlsaSymmetricDistance_gl( ...
                                     'partition', partitionO, ...
                                     'nearestNeighbors', getNNeighbors( obj.sclOutPDistance ) );
            end
            
            % Scaled diffusion operator
            if ~isempty( iSclOutDiffOp )
                if ~isa( varargin{ iSclOutDiffOp }, 'nlsaDiffusionOperator' ) ...
                       && isscalar( varargin{ iSclOutDiffOp } )
                       error( [ msgId 'invalidSclDiffOp' ], ...
                              'The scaled diffusion operator property must be specified as a scalar nlsaDiffusionOperator object' )
                end
                obj.sclOutDiffOp = varargin{ iSclOutDiffOp };
            else
                obj.sclOutDiffOp = nlsaDiffusionOperator_gl( ...
                                  'partition', partitionO, ...
                                  'nEigenfunction', min( 10, nSO ) ); 
            end

            % Scaled projected component
            if ~isempty( iSclOutPrjComponent )
                if ~isa( varargin{ iSclOutPrjComponent }, 'nlsaProjectedComponent' )
                    error( [ msgId 'invalidPrj' ], ...
                        'Projected data must be specified as nlsaProjectedComponent objects' )
                end
                if ~isCompatible( varargin{ iSclOutPrjComponent }, obj.outTrgEmbComponent )
                    error( 'Incompatible projected components' )
                end
                obj.sclOutPrjComponent = varargin{ iSclOutPrjComponent };
            else
                nCT = size( obj.oseTrgEmbComponent, 1 );
                nL  = getNEigenfunction( obj.sclOutDiffOp );
                for iC = nCT : -1 : 1
                    nDE = getEmbeddingSpaceDimension( obj.outTrgEmbComponent( iC ) );
                    obj.sclOutPrjComponent( iC ) = nlsaProjectedComponent( ...
                        'embeddingSpaceDimension', nDE, ...
                        'partition', getPartition( obj.sclPDist ), ...
                        'nBasisFunction', nL );
                end
                obj.sclOutPrjComponent = obj.sclOutPrjComponent';
            end



            % Scaled distances for OS -> IS restriction operator
            if ~isempty( iSclIsrPDistance )
                if ~isa( varargin{ iSclIsrPDistance }, 'nlsaPairwiseDistance_scl' ) ...
                       && ~isscalar( varargin{ iSclIsrPDistance } )
                       error( [ msgId 'invalidSclISrPDist' ], ...
                              'The model pairwise distance property must be specified as a scalar nlsaPairwiseDistance object' )
                end
                obj.sclIsrPDistance = varargin{ iSclIsrPDistance };
            else
                obj.sclIsrPDistance = nlsaPairwiseDistance_scl( ...
                                    'partition', partition, ...
                                    'partitionT', partitionO, ...
                                    'nearestNeighbors', round( nS / 10 ) ); 
            end

            % Model -> nature projection operator
            if ~isempty( iSclIsrDiffOp )
                if ~isa( varargin{ iSclIsrDiffOp }, 'nlsaDiffusionOperator_ose' ) ...
                       && isscalar( varargin{ iSclIsrDiffOp } )
                       error( [ msgId 'invalidIsrSclDiffOp' ], ...
                              'The model diffusion operator property must be specified as a scalar nlsaDiffusionOperator_ose object' )
                end
                obj.sclIsrDiffOp = varargin{ iSclIsrDiffOp };
            else
                obj.sclIsrDiffOp = nlsaDiffusionOperator_ose( ...
                                  'partition', partition, ...
                                  'partitionT', partitionO, ...
                                  'nEigenfunction', min( 10, nSO ) ); 
            end

          
        end
    end
         
    methods( Static )
   
        %% LISTCONSTRUCTORPROPERTIES  List property names for class constructor  
        function pNames = listConstructorProperties
            pNames = nlsaModel_err.listConstructorProperties;
            pNames = [ pNames ...
                      { 'sclOutPairwiseDistance' ...
                        'sclOutSymmetricDistance' ...
                        'sclOutDiffusionOperator' ...
                        'sclOutPrjComponent' ...
                        'sclIsrPairwiseDistance' ...
                        'sclIsrDiffusionOperator' } ];
        end

        %% LISTPARSERPROPERTIES List property names for class constructor parser
        function pNames = listParserProperties
            pNames = nlsaModel_ose.listParserProperties;
            pNames = [ pNames ...
                       { 'sclPairwiseDistanceTemplate' ...
                         'sclSymmetricDistanceTemplate' ...
                         'sclDiffusionOperatorTemplate' ...
                         'modPairwiseDistanceTemplate' ...
                         'modSymmetricDistanceTemplate' ...
                         'modDiffusionOperatorTemplate' ...
                         'modPairwiseDistanceTemplate' ...
                         'modSymmetricDistanceTemplate' ...
                         'prjPairwiseDistanceTemplate' ...
                         'prjDiffusionOperatorTemplate' } ];
        end
            
        %% PARSETEMPLATES  Template parser
        propNameVal = parseTemplates( varargin );
    end    
end
