classdef nlsaKernelDensity_fb < nlsaKernelDensity
%NLSAKERNELDENSITY Class definition and constructor of fixed-bandwidth 
%  kernel density estimation
% 
% Modified 2015/12/15    

    %% PROPERTIES
    properties
        epsilonB    = 2; % base of epsilon
        epsilonELim = [ -40 40 ];
        nEpsilon    = 100; 
        fileQ       = 'dataQ.mat';
        fileDSum    = 'dataDSum.mat';
        pathQ       = 'pathQ';
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKernelDensity_fb( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iEpsilonB    = [];
            iEpsilonELim = [];
            iNEpsilon    = [];
            iFileQ       = [];
            iFileDSum    = [];
            iPathQ       = [];

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
                    case 'densityFile'
                        iFileQ = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'doubleSumFile'
                        iFileDSum = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'densitySubpath'
                        iPathQ = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaKernelDensity( varargin{ ifParentArg } );

            % Set caller-defined values
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
            if ~isempty( iFileQ )
                if ~isrowstr( varargin{ iFileQ } )
                    error( 'Invalid density file specification' )
                end
                obj.fileQ = varargin{ iFileQ };
            end
            if ~isempty( iFileDSum )
                if ~isrowstr( varargin{ iFileDSum } )
                    error( 'Invalid file specification for the double sum' )
                end
                obj.fileDSum = varargin{ iFileDSum };
            end
            if ~isempty( iPathQ )
                if ~isrowstr( varargin{ iPathQ } )
                    error( 'Invalid density subpath specification' )
                end
                obj.pathQ = varargin{ iPathQ };
            end
        end
    end
end    
