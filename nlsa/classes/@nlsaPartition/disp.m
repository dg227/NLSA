function disp( obj, ifVerbose )
% DISP  Display array of nlsaPartition objects
%
% Modified 2012/12/10

if nargin == 1
    ifVerbose = 0;
end
nObj = numel( obj );
for iObj = 1 : nObj
    str = 'NLSA partition object';
    if nObj > 1
        str = [ str, ' (', int2str( iObj ), ')' ];
    end       
    disp( str )
    disp( [ 'Number of samples = ', int2str( getNSample( obj ) ) ] )
    disp( [ 'Number of batches = ', int2str( getNBatch( obj ) ) ] )
    if ifVerbose
        disp( 'Batch limits/sizes:' )
        disp( [ getBatchLimit( obj ), getBatchSize( obj ) ] )
    end
    disp( '' )
end        
