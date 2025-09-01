classdef nlsaEmbeddedComponent_d < nlsaEmbeddedComponent_e
%NLSAEMBEDDEDCOMPONENT_D  Class definition and constructor of differenced
% time lagged embedded data
% 
% Modified 2014/07/10

    methods

    %% CLASS CONSTRUCTOR
        function obj = nlsaEmbeddedComponent_d( varargin )

            nargin = numel( varargin );

            if nargin == 1 && isa( varargin{ 1 }, 'nlsaEmbeddedComponent' ) 
                varargin = { 'dimension', getDimension( varargin{ 1 } ), ...
                             'partition', getPartition( varargin{ 1 } ), ...
                             'origin', getOrigin( varargin{ 1 } ), ...
                             'idxE', getEmbeddingIndices( varargin{ 1 } ), ...
                             'nXB', getNXB( varargin{ 1 } ), ...
                             'nXA', getNXA( varargin{ 1 } ) };
            end
           
            obj = obj@nlsaEmbeddedComponent_e( varargin{ : } );
        end                    
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaEmbeddedComponent_d';
        end

    end
end    
