function setOperator( obj, l, d, q, varargin )
% SETOPERATOR  Set batch data of an nlsaDiffusionOperator_gl object 
%
% Modified 2018/06/17

nS = getNTotalSample( getPartition( obj ) );
ifSave = false( 1, 3 );

if ~isempty( l )
    if any( size( l ) ~= [ nS nS ] )
        error( 'Incompatible operator data' )
    end
    ifSave( 1 ) = true;
end

if ~isempty( d )
    if ~isvector( d ) || numel( d ) ~= nS
        error( 'Incompatible number of elements in degree vector' )
    end
    if isrow( d )
        d = d';
    end
    ifSave( 2 ) = true;
end

if ~isempty( q )
    if ~isvector( q ) || numel( q ) ~= nS
        error( 'Incompatible number of elements in normalization vector' )
    end
    if isrow( q )
        q = q';
    end
    ifSave( 3 ) = true;
end


file = fullfile( getOperatorPath( obj ), getOperatorFile( obj ) );
varNames = { 'l' 'd' 'q' };
save( file, varNames{ ifSave }, varargin{ : } )

