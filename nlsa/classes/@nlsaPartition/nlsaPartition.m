classdef nlsaPartition
% NLSAPARTITION  Class definition and constructor of NLSA partition
%
% Modified 2012/12/10    

    properties
        idx = 1; % end indices of the batches
    end

    methods
        function obj = nlsaPartition( varargin )

            nargin = numel( varargin );

            % Return default object if no arguments
            if nargin == 0
                return
            end

            % Copy nlsaPartition object if in argument list
            if nargin == 1 && isa( varargin{ 1 }, 'nlsaPartition' )
                varargin = { 'idx', getIdx( varargin{ 1 } ) };
                nargin   = numel( varargin );
            end
            
            % Parse input arguments
            iNSample = []; % number of samples
            iNBatch  = []; % number of batches
            iIdx     = []; % explicit batch index specification
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'nSample'
                        iNSample = i + 1;
                    case 'nBatch'
                        iNBatch = i + 1;
                    case 'idx'
                        iIdx = i + 1;
                    otherwise
                        error( [ 'Invalid property ', varargin{ i } ] )
                end
            end 
            if ~isempty( iIdx )
                if isempty( iNSample ) && isempty( iNBatch )
                    idx = varargin{ iIdx };
                    if ~isvector( idx )
                        error( 'Indices must be input in a vector' )
                    end
                    if size( idx, 1 ) > 1
                        idx = idx';
                    end
                    nB  = numel( idx );                    
                    idxRef = 0;
                    for iB = 1 : nB
                        if ~ispsi( idx( iB ) - idxRef )
                            error( 'Invalid index specification' )
                        end
                        idxRef = idx( iB );
                    end
                    obj.idx = idx;
                else
                    error( 'Explicit indices cannot be specified if nS and/or nBatch are specified' )
                end
            else
                if ~isempty( iNSample )
                    nS = varargin{ iNSample };
                    if ~ispsi( nS )
                        error( 'Number of samples must be a positive integer' )
                    end
                else 
                    nS = 1;
                end
                if ~isempty( iNBatch )
                    nB = varargin{ iNBatch };
                    if ~isnnsi( nS - nB )
                        error( 'Number of batches must be a positive integer not exceeding the number of samples' )
                    end
                else
                    nB = 1;
                end
                idx = decomp1d( nS, nB );
                obj.idx = idx( :, 2 )';
            end     
        end
    end
end
