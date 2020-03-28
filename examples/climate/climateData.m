function Data = climateData( dataset, DataSpecs )
% CLIMATEDATA Read climate datasets and output in Matlab format appropriate for
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

switch dataset
case 'noaa'
    if nargout > 0 
        Data = noaaData( DataSpecs );
    else
        noaaData( DataSpecs )
    end
case 'hadisst'
    if nargout > 0
        Data = hadisstData( DataSpecs );
    else
        hadisstData( DataSpecs )
    end
case 'ccsm4'
    if nargout > 0 
        Data = ccsm4Data( DataSpecs );
    else
        ccsm4Data( DataSpecs )
    end
case 'gpcp'
    if nargout > 0 
        Data = gpcpData( DataSpecs );
    else
        gpcpData( DataSpecs )
    end
case 'claus'
    if nargout > 0 
        Data = clausData( DataSpecs );
    else
        clausData( DataSpecs )
    end
otherwise
    error( 'Invalid dataset' )
end


