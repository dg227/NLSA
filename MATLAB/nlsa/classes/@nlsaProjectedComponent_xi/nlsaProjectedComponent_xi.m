classdef nlsaProjectedComponent_xi < nlsaProjectedComponent 
%NLSAPROJECTEDCOMPONENT_XI  Class definition and constructor of data 
% phase space velocity components projected onto an eigenfunction basis
%
% Modified 2014/06/24


    properties
        fileAXi  = 'dataAXi.mat';  % projected velocity data
        pathAXi  = pwd;
    end

    methods

        function obj = nlsaProjectedComponent_xi( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iFileAXi = [];
            iPathAXi = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'velocityProjectionFile'
                        iFileAXi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'velocityProjectionSubpath'
                        iPathAXi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaProjectedComponent( varargin{ ifParentArg } );

            % Set caller defined values
            if ~isempty( iFileAXi )
                if ~isrowstr( varargin{ iFileAXi } )
                    error( 'FileAXi property must be set to a string' )
                end
                obj.fileAXi = varargin{ iFileAXi };
            end
            if ~isempty( iPathAXi )
                if ~isrowstr( varargin{ iPathAXi } )
                    error( 'Invalid state projection path specification' ) 
                end
                obj.pathAXi = varargin{ iPathAXi };
            end
        end                    
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaProjectedComponent_xi';
        end
    end
end    
