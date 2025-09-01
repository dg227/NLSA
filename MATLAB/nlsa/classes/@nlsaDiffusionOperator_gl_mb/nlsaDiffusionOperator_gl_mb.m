classdef nlsaDiffusionOperator_gl_mb < nlsaDiffusionOperator_gl
%NLSADIFFUSIONOPERATOR_GL_MB Class definition and constructor of diffusion 
% operator, implementing the diffusion map algorithm with multiple bandwidths
% 
% Modified 2015/12/16   

    %% PROPERTIES
    properties
        epsilonB    = 2; % bandwidth base
        epsilonELim = [ -40 40 ]; % bandwidth exponents
        nEpsilon    = 100;
        fileDSum   = 'dataDSum.mat';
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaDiffusionOperator_gl_mb( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iEpsilonB     = [];
            iEpsilonELim  = [];
            iNEpsilon     = [];
            iFileDSum     = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'bandwidthBase'
                        iEpsilonB = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'bandwidthExponentLimit'
                        iEpsilonELim = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'nBandwidth'
                        iNEpsilon = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'doubleSumFile'
                        iFileDSum = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaDiffusionOperator_gl( varargin{ ifParentArg } );

            % Set caller defined values
            if ~isempty( iEpsilonB )
                if ~isps( varargin{ iEpsilonB } )
                    error( 'Bandwidth base must be a positive scalar' )
                end
                obj.epsilonB = varargin{ iEpsilonB };
            end
            if ~isempty( iEpsilonELim )
                if ~isvector( numel( varargin{ iEpsilonELim } ) ) ...
                   || numel( varargin{ iEpsilonELim } ) > 2
                   error( 'Invalid exponent limit specificiation' )
                end
                if isrow( numel( varargin{ iEpsilonELim } ) )
                    obj.epsilonELim = varargin{ iEpsilonELim };
                else
                    obj.epsilonELim = varargin{ iEpsilonELim }';
                end
            end
            if ~isempty( iNEpsilon )
                if ~ispsi( varargin{ iNEpsilon } )
                    error( 'Invalid bandwidth number specification' )
                end
                obj.nEpsilon = varargin{ iNEpsilon };
            end
            if ~isempty( iFileDSum )
                if ~isrowstr( varargin{ iFileDSum } )
                    error( 'Invalid file specification for the double sum' )
                end
                obj.fileDSum = varargin{ iFileDSum };
            end
        end
    end

end    
