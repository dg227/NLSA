classdef nlsaLinearMap_gl < nlsaLinearMap
%NLSALINEARMAP Class definition and constructor of linear map with global
% storage format for the temporal patterns
%
% Modified 2015/10/19   

    properties
        fileVT = 'dataVT';
    end

    methods

        function obj = nlsaLinearMap_gl( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iFileVT        = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'temporalPatternFile'
                        iFileVT = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaLinearMap( varargin{ ifParentArg } );

            % Set caller defined values
            if ~isempty( iFileVT )
                if ~isrowstr( varargin{ iFileVT } ) 
                    error( 'Invalid right pattern file specification' )
                end
                obj.fileVT = varargin{ iFileVT };
            end
        end
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaLinearMap_gl';
        end
    end
end    
