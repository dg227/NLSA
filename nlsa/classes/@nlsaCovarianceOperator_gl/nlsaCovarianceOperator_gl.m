classdef nlsaCovarianceOperator_gl < nlsaCovarianceOperator
%NLSACOVARIANCEOPERATOR_GL Class definition and constructor of 
% covariance operator with global storage format
%
% Modified 2014/08/06    

    properties
        fileA    = 'dataA.mat';   % linear map matrix
        fileCV   = 'dataCV.mat';  % right (temporal) covariance operator
        fileCU   = 'dataCU.mat';  % left (spatial) covariance operator 
        fileV    = 'dataV.mat';   % right singular vector file 
   
    end

    methods

        function obj = nlsaCovarianceOperator_gl( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iFileA         = [];
            iFileCV        = [];
            iFileCU        = [];
            iFileV         = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'linearMapFile'
                        iFileA = i + 1;
                        ifParentArg( [ i i + 1 ] ) = true;
                    case 'rightCovarianceFile'
                        iFileCV = i + 1;
                        ifParentArg( [ i i + 1 ] ) = true;
                    case 'leftCovarianceFile'
                        iFileCU = i + 1;
                        ifParentArg( [ i i + 1 ] ) = true;
                    case 'rightSingularVectorFile'
                        iFileV = i + 1;
                        ifParentArg( [ i i + 1 ] ) = true;
                end
            end

            obj = obj@nlsaCovarianceOperator( varargin{ ifParentArg } );

            % Set caller defined values
            if ~isempty( iFileA )
                if ~isrowstr( varargin{ iFileA } ) 
                    error( 'Invalid linear map file specification' )
                end
                obj.fileA = varargin{ iFileA };
            end
            if ~isempty( iFileCV )
                if ~isrowstr( varargin{ iFileCV } ) 
                    error( 'Invalid right covariance matrix file specification' )
                end
                obj.fileCV = varargin{ iFileCV };
            end
            if ~isempty( iFileCU )
                if ~isrowstr( varargin{ iFileCU } ) 
                    error( 'Invalid left covariance matrix file specification' )
                end
                obj.fileCU = varargin{ iFileCU };
            end
            if ~isempty( iFileV )
                if ~isrowstr( varargin{ iFileV } ) 
                    error( 'Invalid right singular vector file specification' )
                end
                obj.fileV = varargin{ iFileV };
            end
        end
    end
end    
