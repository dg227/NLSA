classdef nlsaEmbeddedComponent_ose_n < nlsaEmbeddedComponent_ose
%NLSAEMBEDDEDCOMPONENT_OSE_N  Class definition and constructor of NLSA 
%  time-lagged embedded component with out-of-sample extension for phase
%  space velocity data using the Nystrom method (projection onto 
%  eigenfunction basis)
%
%  Modified 2020/07/25


    properties
        idxPhi = 1;
    end

    methods

    %% CLASS CONSTRUCTOR
        function obj = nlsaEmbeddedComponent_ose_n( varargin )

            nargin = numel( varargin );

            if nargin == 1 
                if isa( varargin{ 1 }, 'nlsaEmbeddedComponent_xi' )  
                    varargin = { ...
                        'dimension', getDimension( varargin{ 1 } ), ...
                        'partition', getPartition( varargin{ 1 } ), ...
                        'origin', getOrigin( varargin{ 1 } ), ...
                        'idxE', getEmbeddingIndices( varargin{ 1 } ), ...
                        'nXB', getNXB( varargin{ 1 } ), ...
                        'nXA', getNXA( varargin{ 1 } ), ...
                        'fdOrder', getFDOrder( varargin{ 1 } ), ...
                        'fdType', getFDType( varargin{ 1 } ), ...
                        'componentTag', getComponentTag( varargin{ 1 } ), ...
                        'realizationTag', getRealizationTag( varargin{ 1 } ) };

                elseif isa( varargin{ 1 }, 'nlsaEmbeddedComponent' ) 
                    varargin = { ...
                        'dimension', getDimension( varargin{ 1 } ), ...
                        'partition', getPartition( varargin{ 1 } ), ...
                        'origin', getOrigin( varargin{ 1 } ), ...
                        'idxE', getEmbeddingIndices( varargin{ 1 } ), ...
                        'nXB', getNXB( varargin{ 1 } ), ...
                        'nXA', getNXA( varargin{ 1 } ), ...
                        'componentTag', getComponentTag( varargin{ 1 } ), ...
                        'realizationTag', getRealizationTag( varargin{ 1 } ) };
                end
                nargin = numel( varargin );
            end
            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iIdxPhi = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'eigenfunctionIdx'
                        iIdxPhi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'eigenfunctionTag'
                        iTagPhi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaEmbeddedComponent_ose( varargin{ ifParentArg } );

            % Set caller-defined values
            if ~isempty( iIdxPhi )
                if ~obj.isValidIdx( varargin{ iIdxPhi } )
                        error( 'Invalid basis function index specification' )
                end
                obj.idxPhi = varargin{ iIdxPhi };
            end
       end                    
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaEmbeddedComponent_ose_n';
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
            idxRef = 0; 
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
