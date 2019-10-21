classdef nlsaLocalDistanceData
%NLSALOCALDISTANCE_DATA   Data container for local distances
%
%
%   Modified 2015/10/28

    properties
        component = nlsaComponent();
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaLocalDistanceData( varargin )
            
            msgId = 'nlsaLocalDistanceData:';

            % Return default object if no input arguments
            if nargin == 0
                return
            end

            % Parse input arguments
            iData = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'component'
                        iData = i + 1;
                    otherwise
                        error( [ msgId 'invalidProperty' ], ...
                               [ 'Invalid property name ' varargin{ i } ] ) 
                end
            end 

            % Set caller-defined values

            % Source components 
            if ~isempty( iData )
                if ~isa( varargin{ iData }, 'nlsaComponent' )
                    error( [ msgId 'invalidData' ], ...
                           'Source data must be specified as an array of nlsaComponent objects.' )
                end
            else
                error( [ msgId 'emptyData' ], 'Unassigned data' )
            end
            [ ifC, Test ] = isCompatible( varargin{ iData } ); 
            if ~ifC
                disp( Test )
                error( [ msgId 'incompatibleData' ], 'Incompatible data' )
            end
            obj.component = varargin{ iData };     
        end
    end
end
