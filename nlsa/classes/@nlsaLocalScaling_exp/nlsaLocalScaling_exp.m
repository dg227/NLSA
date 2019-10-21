classdef nlsaLocalScaling_exp < nlsaLocalScaling
%NLSALOCALSCALING_EXP  Class definition and constructor of local distance 
% scaling by an exponential factor
%
% Modified 2014/06/17

    %% PROPERTIES
    properties
        pX   = 2; % exponent for state error
        pXi  = 2; % exponent for phase space velocity error
        bX   = 1; % proportionality constant for state error
        bXi  = 1; % proportionality constant for velocity error
    end

    methods
        %% NLSALOCALSCALING_EXP  Class constructor
        function obj = nlsaLocalScaling_exp( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iBX  = [];
            iBXi = [];
            iPX  = [];
            iPXi = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'propX'
                        iBX = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'propXi'
                        iBXi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'pwrX'
                        iPX = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'pwrXi'
                        iPXi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaLocalScaling( varargin{ ifParentArg } );
           
            % Set caller-defined values
            if ~isempty( iBX )
                if ~isscalar( varargin{ iBX } )
                    error( 'State proportionaility constant must be a scalar' )
                end
                obj.bX = varargin{ iBX };
            end
            if ~isempty( iBXi )
                if ~isscalar( varargin{ iBXi } )
                    error( 'Velocity proportionaility constant must be a scalar' )
                end
                obj.bXi = varargin{ iBXi };
            end
            if ~isempty( iPX )
                if ~isscalar( varargin{ iPX } )
                    error( 'State error exponent must be a scalar' )
                end
                obj.pX = varargin{ iPX };
            end
            if ~isempty( iPXi )
                if ~isscalar( varargin{ iPXi } )
                    error( 'Velocity error exponent must be a scalar' )
                end
                obj.pXi = varargin{ iPXi };
            end
        end
    end
end
