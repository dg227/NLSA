classdef nlsaLinearMap < nlsaCovarianceOperator_gl
%NLSALINEARMAP Class definition and constructor of linear map
%
% Modified 2015/10/19  

    properties
        idxPhi = 1;         % eigenfunction indices
        pathVT = 'dataVT';  % temporal patterns subdirectory
    end

    methods

        function obj = nlsaLinearMap( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iIdxPhi        = [];
            iPathVT        = [];
            iNeig          = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'basisFunctionIdx'
                        iIdxPhi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'temporalPatternSubpath'
                        iPathVT = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'nEigenfunction'
                        iNEig = i + 1;
                end
            end

            obj = obj@nlsaCovarianceOperator_gl( varargin{ ifParentArg } );

            % Set caller defined values
            if ~isempty( iIdxPhi ) 
                if ~isempty( iNeig )
                    error( 'Eigenfunction indices and number of eigenfunctions cannot be specified simultaneously' )
                end
                if ~obj.isValidIdx( varargin{ iIdxPhi } ) 
                    error( 'Invalid basis function index specification' )
                end
                obj.idxPhi = varargin{ iIdxPhi };
                obj = setNEigenfunction( obj, numel( varargin{ iIdxPhi } ) );
            else 
                obj.idxPhi = 1 : getNEigenfunction( obj );
            end
            if ~isempty( iPathVT )
                if ~isrowstr( varargin{ iPathVT } ) 
                    error( 'Invalid temporal pattern subpath specification' )
                end
                obj.pathVT = varargin{ iPathVT };
            end
        end
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaLinearMap';
        end

        %% ISVALIDIDX Helper function to validate basis function indices 
        function ifV = isValidIdx( idx )
            ifV = true;

            if ~isvector( idx )
                ifV = false;
                return
            end

            idx  = sort( idx, 'ascend' );
            nPhi = numel( idx );
            idxRef = -1; 
            for iPhi = 1 : nPhi
                if ~ispsi( idx( iPhi ) - idxRef )
                    ifV = false;
                    return
                end
                idxRef = idx( iPhi );
            end
        end
    end
end    
