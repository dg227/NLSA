classdef nlsaEmbeddedComponent_rec < nlsaEmbeddedComponent_xi_e
%NLSAEMBEDDEDCOMPONENT_REC  Class definition and constructor of NLSA 
% time-lagged embedded component reconstructed from eigenfunction projected
% data
%
%
% Modified 2014/08/04


    properties
        idxPhi = 1;
        tagPhi = 1; % basis function tag
    end

    methods

    %% CLASS CONSTRUCTOR
        function obj = nlsaEmbeddedComponent_rec( varargin )

            nargin = numel( varargin );

            if nargin == 1 && isa( varargin{ 1 }, 'nlsaEmbeddedComponent_xi' ) 
                varargin = { 'dimension', getDimension( varargin{ 1 } ), ...
                             'partition', getPartition( varargin{ 1 } ), ...
                             'origin', getOrigin( varargin{ 1 } ), ...
                             'idxE', getEmbeddingIndices( varargin{ 1 } ), ...
                             'nXB', getNXB( varargin{ 1 } ), ...
                             'nXA', getNXA( varargin{ 1 } ), ...
                             'fdOrder', getFDOrder( varargin{ 1 } ), ...
                             'fdType', getFDType( varargin{ 1 } ), ...
                             'componentTag', getComponentTag( varargin{ 1 } ), ...
                             'realizationTag', getRealizationTag( varargin{ 1 } ) };

                nargin = numel( varargin );
            end
            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iIdxPhi = [];
            iTagPhi = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'basisFunctionIdx'
                        iIdxPhi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'basisFunctionTag'
                        iTagPhi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaEmbeddedComponent_xi_e( varargin{ ifParentArg } );

            % Set caller-defined values
            if ~isempty( iIdxPhi )
                if ~obj.isValidIdx( varargin{ iIdxPhi } )
                        error( 'Invalid basis function index specification' )
                end
                obj.idxPhi = varargin{ iIdxPhi };
            end

            if ~isempty( iTagPhi )
                if ~isrowstr( varargin{ iTagPhi } ) 
                    error( 'Invalid embedding tag' )
                end
                obj.tagPhi = varargin{ iTagPhi };
            end 
       end                    
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaEmbeddedComponent_rec';
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
