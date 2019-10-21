classdef nlsaLocalDistanceFunction
%NLSAPAIRWISEDISTANCE  Class definition and constructor of nlsaPairwiseDistance
% objects
%
% Modified 2015/10/29   

   %% PROPERTIES
    properties
        lDistance     = nlsaLocalDistance_l2();
        Q             = struct(); % query data 
        T             = struct(); % test data
    end

    methods

        %% NLSAPAIRWISEDISTANCE  Class contructor
        function obj = nlsaLocalDistanceFunction( varargin )


            % Parse input arguments
            iLDistance     = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'localDistance'
                        iLDistance = i + 1;
                    otherwise
                        error( 'Invalid property' )
                end
            end

            if ~isempty( iLDistance )
                if ~isa( varargin{ iLDistance }, 'nlsaLocalDistance' )
                    error( 'Local distance property must be set to an nlsaLocalDistance object' )
                end
                obj.lDistance = varargin{ iLDistance };
            end
        end
    end
end    
