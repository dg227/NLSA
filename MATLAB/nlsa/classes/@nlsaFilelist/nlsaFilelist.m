classdef nlsaFilelist
%NLSAFILELIST  Helper class for managing file lists
%
% Modified 2014/06/13 

    %% PROPERTIES
    properties
        nF   = 1;
        file = { '' };
    end

    methods

        %% NLSAFILELIST  Class constructor
        function obj = nlsaFilelist( varargin )

            if nargin == 1 && isa( varargin{ 1 }, 'nlsaPartition' )
                % Build nlsaFilelist objects compatible with an array of
                % nlsaPartition objects
                for iObj = numel( varargin{ 1 } ) : -1 : 1
                     obj( iObj ) = nlsaFilelist( ...
                                 'nFile', getNBatch( varargin{ 1 }( iObj ) ) ); 
                end
                return
            end
            
            % Parse input arguments
            iNF    = [];
            iFile = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'file'
                        iFile = i + 1;
                    case 'nFile'
                        iNF = i + 1;
               end
            end
            

            % Set caller-defined values
            if ~isempty( iNF )
                if ~ispsi( varargin{ iNF } )
                    error( 'Number of files must be a positive scalar integer' )
                end
                obj.nF = varargin{ iNF };
            end
            if ~isempty( iFile )
                if isrowstr( varargin{ iFile } )
                    varargin{ iFile } = { varargin{ iFile } };
                end
                if   ~iscellstr( varargin{ iFile } ) ...
                  || ~isrow( varargin{ iFile } ) ...
                  || numel( varargin{ iFile } ) ~= obj.nF
                    error( 'Files must be specified as a cell row vector of strings of size equal to the number of files' )
                end
                obj.file = varargin{ iFile };
            else
                obj.file = cell( 1, obj.nF );
                for iF = 1 : obj.nF
                    obj.file{ iF } = '';
                end
            end
        end
    end
end    
