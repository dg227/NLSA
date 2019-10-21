classdef nlsaDiffusionOperator_batch < nlsaDiffusionOperator
%NLSADIFFUSIONOPERATOR_BATCH Class definition and constructor of diffusion 
%  operator with batch storage format
% 
% Modified 2018/06/18  

    properties
        nN       = 2; 
        nNT      = [];                   % nearest neighbors used when summing over test data; reverts to nN if empty
        filePhi  = nlsaFilelist();       % eigenfunction file
        fileP    = nlsaFilelist();       % operator file
        fileQ    = nlsaFilelist();       % normalization file
        fileD    = nlsaFilelist();       % degree file
        pathQ    = 'dataQ';             
        pathD    = 'dataD'; 
        precEigs = 'double';             % precision to solve the eigenvalue problem
    end

    methods

        function obj = nlsaDiffusionOperator_batch( varargin )

            nargin = numel( varargin );

            if nargin == 1
                if isa( varargin{ 1 }, 'nlsaDiffusionOperator' )
                
                    varargin = { 'alpha',             getAlpha( varargin{ 1 } ), ...
                                 'epsilon',           getEpsilon( varargin{ 1 } ), ...
                                 'nEigenfunction',    getNEigenfunction( varargin{ 1 } ), ...
                                 'path', getPath( varargin{ 1 } ), ...
                                 'operatorSubpath', getOperatorSubpath( varargin{ 1 } ), ...
                                 'eigenfunctionSubpath', getEigenfunctionSubpath( varargin{ 1 } ), ...
                                 'eigenvalueFile', getEigenvalueFile( varargin{ 1 } ), ...
                                 'tag', getTag( varargin{ 1 } ) };

                    nargin = numel( varargin );
                end
            end

            ifParentArg    = true( 1, nargin );
            
            % Parse input arguments
            iNN        = [];
            iNNT       = [];
            iFilePhi   = [];
            iFileP     = [];
            iFileQ     = [];
            iFileD     = [];
            iPathQ     = [];
            iPathD     = [];
            iPrecEigs  = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'eigenfunctionFile'
                        iFilePhi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'operatorFile'
                        iFileP = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false; 
                    case 'nNeighbors'
                        iNN = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'nNeighborsT'
                        iNNT = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'normalizationFile'
                        iFileQ = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'degreeFile'
                        iFileD = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'normalizationSubpath'
                        iPathQ = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'degreeSubpath'
                        iPathD = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'precisionEigs'
                        iPrecEigs = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaDiffusionOperator( varargin{ ifParentArg } );
             
            % Set caller defined values
            partition  = getPartition( obj );
            partitionT = getPartitionTest( obj );
            if ~isempty( iFileP )
                if ~isa( varargin{ iFileP }, 'nlsaFilelist' ) ...
                  || ~isCompatible( varargin{ iFileP }, partition )
                    error( 'Operator file property must be set to a vector of nlsaFilelist objects compatible with the query partition' )
                end
                obj.fileP = varargin{ iFileP };
            else
                obj.fileP = nlsaFilelist( partition );
            end
            if ~isempty( iFilePhi )
                if ~isa( varargin{ iFilePhi }, 'nlsaFilelist' ) ...
                  || ~isCompatible( varargin{ iFilePhi }, partition )
                    error( 'Eigenfunction file property must be set to a vector of nlsaFilelist objects compatible with the query partition' )
                end
                obj.filePhi = varargin{ iFilePhi };
            else
                obj.filePhi = nlsaFilelist( partition );
            end
            if ~isempty( iFileQ )
                if ~isa( varargin{ iFileQ }, 'nlsaFilelist' ) ...
                  || ~isCompaticle( varargin{ iFileQ }, partition ) 
                    error( 'Normalization file property must be set to a vector of nlsaFilelist objects compatible to the query or test partitions' )
                end
                obj.fileQ = varargin{ iFileQ };
            else
                obj.fileQ = nlsaFilelist( partition );
            end
            if ~isempty( iFileD )
                 if ~isa( varargin{ iFileD }, 'nlsaFilelist' ) ...
                   || ~isCompaticle( varargin{ iFileD }, partitionT ) ...
                   || ~isCompatible( varargin{ iFileD }, partition ) 
                     error( 'Degree file property must be set to a vector of nlsaFilelist objects compatible to the query or test partitions' )
                 end
                 obj.fileD = varargin{ iFileD };
            else
                 obj.fileD = nlsaFilelist( partition );
            end
            if ~isempty( iNN )
                if ~ispsi( varargin{ iNN } ) 
                    error( 'Number of nearest neighbors must be a positive scalar integer' )
                end
                obj.nN = varargin{ iNN };
            end
            if ~isempty( iNNT )
                if ~ispsi( varargin{ iNNT } ) 
                    error( 'Number of nearest neighbors for test data must be a positive scalar integer' )
                end
                obj.nNT = varargin{ iNNT };
            end
            if ~isempty( iPathQ )
               if ~isrowstr( varargin{ iPathQ } )
                   error( 'Invalid normalization subpath specification' )
               end
               obj.pathQ = varargin{ iPathQ };
            end
            if ~isempty( iPathD )                
                if ~isrowstr( varargin{ iPathD } )
                    error( 'Invalid degree subpath specification' )
                end
                obj.pathD = varargin{ iPathD };
            end
            if ~isempty( iPrecEigs )
                if ~isrowstr( varargin{ iPrecEigs } ) ...
                  || ~any( strcmp( varargin{ iPrecEigs }, { 'single' 'double' } ) )
                    error( 'Invalid precision specification' )
                end
                obj.precEigs = varargin{ iPrecEigs };
            end 
        end
    end
end    
