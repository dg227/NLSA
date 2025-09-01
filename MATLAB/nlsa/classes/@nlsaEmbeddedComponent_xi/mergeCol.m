function obj = mergeCol( obj, src, varargin )
%% MERGECOL Merge a matrix of nlsaComponent_xi objects to a column vector 
% 
% This method differs from the corresponding method of the nlsaComponent class
% in that it updates the fileXi property. 
%
% Modified 2020/04/03

% Check input sizes
if ~iscolumn( obj ) 
    error( 'First argument must be a column vector' )
end

nC = numel( obj );
if ~( ismatrix( src ) && size( src, 1 ) == nC )
    error( 'Second argument must be a matrix with equal number of rows to the number of elements of the first argument' )
end

if ~isCompatible( src )
    error( 'Incompatible input components' )
end

% Default partition, files, and tags
defPartition = mergePartitions( getPartition( src( 1, : ) ) ); 
defFilelist  = nlsaFilelist( 'nFile', getNBatch( defPartition ) );
defTagR      = concatenateRealizationTags( src ); 

if nargin == 2
    % Default partition and realization tags
    obj = duplicate( obj, src( :, 1 ), ...
        'partition', defPartition, 'file', defFilelist, ...
        'fileXi', defFilelist, 'tagR', defTagR );
else
    % Optional input arguments passed as property name-value pairs
    [ tst, props ] = isPropNameVal( varargin{ : } );
    if tst
        if ~any( strcmp( props, 'partition' ) )
             varargin = [ varargin { 'partition' defPartition } ]; 
             if ~any( strcmp( props, 'file' ) )
                 varargin = [ varargin { 'file' defFilelist ...
                     'fileXi', defFilelist } ];
             end
        else
            iPartition = find( strcmp( props, 'partition' ) );
            if ~isFiner( defPartition, varargin{ 2 * iPartition  } )
                error( 'Incompatible partitions' )
            end
            if ~any( strcmp( props, 'file' ) )
                filelist = nlsaFilelist( ...
                    'nFile', getNBatch( varargin{ 2 * iPartition } ) );
                varargin = [ varargin { 'file' filelist, ...
                            'fileXi' filelist } ];
            else 
                iFile = find( strcmp( props, 'file' ) );
                if getNFile( varargin{ 2 * iFile } ) ~= ...
                   getNBatch( varargin{ 2 * iParttition } )
                     error( 'Incompatible filelist.' )
                end
            end
        end
        if ~any( strcmp( props, 'tagR' ) )
            varargin = [ varargin { 'tagR' defTagR } ];
        end
        obj = duplicate( obj, src( :, 1 ), varargin{ : } );
     else
         error( [ 'Optional input arguments must be entered as property ' ...
                  'name-value pairs' ] )
     end
end


