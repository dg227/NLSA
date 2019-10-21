classdef nlsaEmbeddedComponent_xi_d < nlsaEmbeddedComponent_d ...
                                    & nlsaEmbeddedComponent_xi_e
%NLSAEMBEDDEDCOMPONENT_XI_D  Class definition and constructor of differenced
% time lagged embedded data with phase space velocity
%
% Modified 2014/09/02

    methods

    %% CLASS CONSTRUCTOR
        function obj = nlsaEmbeddedComponent_xi_d( varargin )

            nargin = numel( varargin );

            if nargin == 1 && isa( varargin{ 1 }, 'nlsaEmbeddedComponent_xi' ) 
                varargin = { 'dimension', getDimension( varargin{ 1 } ), ...
                             'partition', getPartition( varargin{ 1 } ), ...
                             'origin', getOrigin( varargin{ 1 } ), ...
                             'idxE', getEmbeddingIndices( varargin{ 1 } ), ...
                             'nXB', getNXB( varargin{ 1 } ), ...
                             'nXA', getNXA( varargin{ 1 } ), ...
                             'fdOrder', getFDOrder( varargin{ 1 } ), ...
                             'fdType', getFDType( varargin{ 1 } ) };

            end
            obj@nlsaEmbeddedComponent_d(); 
            obj@nlsaEmbeddedComponent_xi_e( varargin{ : } );
        end                    
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaEmbeddedComponent_xi_d';
        end

    end
end    
