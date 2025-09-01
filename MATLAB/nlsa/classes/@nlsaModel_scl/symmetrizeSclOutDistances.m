function symmetrizeSclOutDistances( obj )
% SYMMETRIZESCLOUTDISTANCES Compute symmetric scaled distance matrix from 
% pairwise distances for the OS data of an nlsaModel_scl object
% 
% Modified 2014/07/28

sDistance = getSclOutSymmetricDistance( obj );
pDistance = getSclOutPairwiseDistance( obj );

logFile = 'dataYS.log'; 
symmetrizeDistances( sDistance, pDistance, ...
                    'logPath', getDistancePath( sDistance ), ...
                    'logFile', logFile );
