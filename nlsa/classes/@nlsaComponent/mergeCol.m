function obj = mergeCol( obj, src, varargin )
%% MERGE_COL Merge a matrix of nlsaComponent objects to a column vector 
%
% Modified 2019/11/16

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

% Degault partition and tags
defPartition = mergePartitions( getPartition( src( 1, : ) ) );
defTagR      = concatenateRealizationTags( src ); 

if nargin == 2
    % Default partition and realization tags
    obj = duplicate( obj, src( :, 1 ), ...
        'partition', defPartition, 'tagR', defTagR );
 else
     % Optional input arguments passed as property name-value pairs
     [ tst, props ] = isPropNameVal( varargin{ : } );
     if tst
         if ~any( strcmp( props, 'partition' ) )
             varargin = [ varargin 'partition' defPartition ]; 
         else
            iPartition = find( strcmp(  props, 'partition' ) );
            if ~isFiner( defPartition, varargin{ 2 * iPartition  } )
                error( 'Incompatible partitions' )
            end
end
         if ~any( strcmp( props, 'tagR' ) )
             varargin = [ varargin { 'tagR' } { defTagR } ];
         end
         obj = duplicate( obj, src( :, 1 ), varargin{ : } );
     else
         error( 'Optional input arguments must be entered as property name-value pairs' )
     end
 end


