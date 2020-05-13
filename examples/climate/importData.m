function Data = importData( dataset, DataSpecs )
% IMPORTDATA Read climate datasets and output in Matlab format appropriate for
% NLSA code. 
%
% Input arguments:
%
% dataset:      A string identifier for the dataset. Possible options are:
%
%               - noaa    (NOAA 20th century reanalysis)
%               - hadisst (HadISST dataset) 
%               - ccsm4   (CCSM4 model)
%               - claus   (CLAUS brightness temperature dataset)
%               - gpcp    (GPCP griddded precipitation dataset)
%
% DataSpecs:    A data structure containing data specifications, such as 
%               input filenames, spatial domain, and time domain. 
%
% DataSpecs is passed to a lower-level function implementing data drivial 
% for the chosen dataset 
% 
% Output arguments:
%
% Data:         A data structure containing the data retrived along with 
%               associated attributes.
%
% Modified 2020/05/12

switch dataset
case 'noaa'
    if nargout > 0 
        Data = importData_noaa( DataSpecs );
    else
        importData_noaa( DataSpecs )
    end
case 'hadisst'
    if nargout > 0
        Data = importData_hadisst( DataSpecs );
    else
        importData_hadisst( DataSpecs )
    end
case 'ccsm4Ctrl'
    if nargout > 0 
        Data = importData_ccsm4Ctrl( DataSpecs );
    else
        importData_ccsm4Ctrl( DataSpecs )
    end
case 'gpcp'
    if nargout > 0 
        Data = importData_gpcp( DataSpecs );
    else
        importData_gpcp( DataSpecs )
    end
case 'claus'
    if nargout > 0 
        Data = importData_claus( DataSpecs );
    else
        importData_claus( DataSpecs )
    end
otherwise
    error( 'Invalid dataset' )
end


