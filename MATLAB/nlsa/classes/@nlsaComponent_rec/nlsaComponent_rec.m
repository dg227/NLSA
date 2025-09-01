classdef nlsaComponent_rec < nlsaComponent
%NLSACOMPONENT_REC  Class definition and constructor of NLSA component 
% from eigenfunction projected data
%
% Modified 2015/12/08


    methods

    %% CLASS CONSTRUCTOR
        function obj = nlsaComponent_rec( varargin )

            nargin = numel( varargin );

            if nargin == 1 && isa( varargin{ 1 }, 'nlsaComponent' ) 
                varargin = { 'dimension', getDimension( varargin{ 1 } ), ...
                             'partition', getPartition( varargin{ 1 } ), ...
                             'componentTag', getComponentTag( varargin{ 1 } ), ...
                             'realizationTag', getRealizationTag( varargin{ 1 } ) };

                nargin = numel( varargin );
            end

            obj = obj@nlsaComponent( varargin{ : } );
       end                    
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaComponent_rec';
        end
    end
end    
