function [ ifC, Test ] = isCompatible_comp( obj )
% ISCOMPATIBLE_COMP Check compatibility of nlsaComponent objects 
%
% Modified 2014/02/03

[ nC1, nR1 ]        = size( obj );
Test.partition     = mergePartitions( obj( 1, : ) );
Test.passPartition = true;
Test.nD            = getDimension( obj( :, 1 ) );
Test.passND        = true;
ifC                 = true; 


% Check that partitions are compatible for each component
for iC = 2 : nC1
    if ~isequal( Test.partition, mergePartitions( obj( iC, : ) ) );
        Test.passPartition = false;
        ifC                 = false;
        break
    end
end


% Check that dimensions are compatible for each realization	
for iR = 2 : nR1
    if any( Test.nD ~= getDimension( obj( :, iR ) ) )
        Test.passND  = false;
        ifC           = false;
        break
    end
end

