classdef nlsaLocalScaling_pwr < nlsaLocalScaling_exp
%NLSALOCALSCALING_PWR  Class definition and constructor of local distance 
% scaling by a shifted power-law factor
%
% Modified 2014/06/18

    %% PROPERTIES
    properties
        p    = 1; % exponent
        c    = 1; % shift constant 
    end

    methods
        %% NLSALOCALSCALING_PWR  Class constructor
        function obj = nlsaLocalScaling_pwr( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iC   = [];
            iP   = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'const'
                        iC = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'pwr'
                        iP = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaLocalScaling_exp( varargin{ ifParentArg } );
           
            % Set caller-defined values
            if ~isempty( iC )
                if ~isscalar( varargin{ iC } )
                    error( 'Constant must be a scalar' )
                end
                obj.c = varargin{ iC };
            end
            if ~isempty( iP )
                if ~isscalar( varargin{ iP } )
                    error( 'Exponent parameter must be a scalar' )
                end
                obj.p = varargin{ iP };
            end
        end
    end
end
