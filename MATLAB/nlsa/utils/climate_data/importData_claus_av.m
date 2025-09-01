%% This program reads in one binary ClAUS portable gray map 
%% Input data is low resolution 0.5 deg x 0.5 deg version, Lon = 720 pts, Lat = 359 pts
% The (unsigned) byte data values in the brightness temperature files are mapped so that the values
% b=1 to 255 correspond to TB=340 to 170 K linearly, i.e.:
% TB = 340 - (b-1)*170/254
% Missing value:  0  
% The temperature resolution of the data is thus approximately 0.67 K.
% % The longitude of the leftmost column corresponds to 0° 
% %The top row of the lo_res dataset corresponds to 89° 30' N
% the truncated data domain is (15S?15N, 80E?160W) (locations in original data: r150-210,c161-401)
% Linear interp in time is used to fill the missing values completely.
%
% Modified 2017/11/10

dataDirIn  = '/kontiki_array5/data/claus/lo_res';
experiment = 'lo_res';                  
lbl        = 'tb';                          % brightness temperature
xLim       = [ 0 359.5  ];                  % longitude limits
yLim       = [ -15 15 ];                   % lattitude limits
res        = .5;                            % resolution
%tLim       = { '1983070100' '1983073121' }; % time limits 
tLim       = { '1983070100' '1983070221' }; % time limits 
nDX        = 2;                             % downsampling in x direction
nDY        = 2;                             % downsampling in y direction
nDT        = 2;                             % downsampling in time
avg        = 'sym';

switch avg
    case 'sym'
        lbl = [ lbl '_lat_av' ];
    case 'asym'
        lbl = [ lbl '_lat_aav' ];
end

str = [ 'x',   int2str( xLim( 1 ) ), '-', int2str( xLim( 2 ) ), ...
        '_y',  int2str( yLim( 1 ) ), '-', int2str( yLim( 2 ) ), ...
        '_',  tLim{ 1 }, '-', tLim{ 2 }, ...
        '_nDX', int2str( nDX ), ...
        '_nDY', int2str( nDY ), ...
        '_nDT', int2str( nDT ) ];

dataDir = fullfile( 'data', 'raw', lbl, str );

if ~isdir( dataDir )
    mkdir( dataDir )
end

t     = datenum( tLim{ 1 }, 'yyyymmddhh' ) : 1/8 : datenum( tLim{ 2 }, 'yyyymmddhh' );
iXLim = xLim / res + 1;                       % columns of cropped domain
iYLim = ( 89.5 - yLim( [ 2 1 ] ) ) / res + 1; % rows of cropped domain
nX    = ( iXLim( 2 ) - iXLim( 1 ) ) + 1;
nY    = ( iYLim( 2 ) - iYLim( 1 ) ) + 1; 
nXY   = nX * nY;
nT    = numel( t );
nxo   = 720; % original longitude size
nyo   = 359; % original lattitude size
undef = 0;   % undefined values

tic

myData  = zeros( nXY, nT ) ; % preallocating space

for nf=1:nT
    cc=datestr(t(nf),30);
    dl1=cc(1:4);
    dl2=cc(1:6);
    dl3=strcat(cc(1:8),cc(10:11));
    filename=strcat( dataDirIn, '/', dl1,'/',dl2,'/',dl3,'.2bt'); 
    %disp( filename )
    I = importdata(filename);
    myData( :, nf ) = reshape( double( I( iYLim( 2 ) : -1 : iYLim( 1 ), ...
                                          iXLim( 1 ) :  1 : iXLim( 2 ) ) ), [ nXY 1 ] ); % Crop sub-domain, flip to be S-N, convert to double
    clear I
end
disp('finished reading files')

myData(bsxfun(@le, myData,undef))=NaN;% undefined values are treated as NaN; useful for interp

for nrow=1:nXY
    x=myData(nrow,:);
    y=x;
    miss_range=isnan(x);
    ref_range=find(~miss_range);
    miss_range([1:(min(ref_range)-1) (max(ref_range)+1):end])=0; % if missing values are at the boundaries, hopeless anyway so revert to zero
    y(miss_range)=interp1(ref_range,x(ref_range),find(miss_range));
    myData(nrow,:)=y;
end

    CHK=sum(sum(isnan(myData))); % checked to be zero, so no spatial interp is needed.

sprintf('finished interpolation, missing points= %d out of %d elements', CHK, nXY * nT )
clear x; clear y; clear miss_range; clear ref_range; clear CHK;

myData = 340 - ( myData - 1 ) * ( 170 / 254 ); % convert to brightness temperature

toc

x = xLim( 1 ) : res : xLim( 2 );
y = yLim( 1 ) : res : yLim( 2 ); 
w = ones( nXY, 1 );                % uniform area weights

if any( w ~= 1 )
    disp( 'Rescaling by grid areas...' )
    myData = bsxfun( @times, myData, w );
end

% Downsample and remove NANs
x   = x( 1 : nDX : end );
y   = y( 1 : nDY : end );
t   = t( 1 : nDT : end );
nT = numel( t );
myData  = myData( :, 1 : nDT : end );
myData  = reshape( myData, [ nY nX nT ] );
myData  = myData( 1 : nDY : end, 1 : nDX : end, : );
w = reshape( w, [ nY nX ] );
w = w( 1 : nDY : end, 1 : nDX : end );


nX  = numel( x );
nY  = numel( y );
nXY = nX * nY;

switch avg
    case 'sym'
        myData2 = zeros( nX, nT );
        for iT = 1 : nT
            myData2( :, iT ) = mean( myData( :, iT ), 1 )';
        end
        
        myData = myData2;
        clear myData2;
        disp( 'finished symmetric latitude averaging' )
    case 'asym'
        SH=bsxfun(@lt,y',0);
        NH=bsxfun(@gt,y',0);
        myData2 = zeros( nX, nT );
        for iT = 1 : nT
        tempData= squeeze( myData( :, :, iT ) );
        myData2( :, iT ) = mean( ( tempData(SH,:)-flipud(tempData(NH,:)) ), 1 )';
        end
        myData = myData2;
        clear myData2; clear tempData
        disp( 'finished antisymmetric latitude averaging' )
end


mu     = mean( myData, 2 );
ifDef = ~isnan( mu );
nD    = nnz( ifDef );

mu = mu( ifDef )';
w  = w( ifDef )';
myData2 = myData;
myData  = zeros( nD, nT ); 
for iT = 1 : nT
    myDataTmp       = myData2( :, iT );
    myData( :, iT ) = myDataTmp( ifDef );
end 
clear myData2

l2NormT = sum( myData .^ 2, 1 );
l2Norm  = sqrt( sum( l2NormT ) / nT );
l2NormT = sqrt( l2NormT ); 

dataFile = fullfile( dataDir, 'dataGrid.mat' );
save( dataFile, 'x', 'y', 'w', 'lbl', 'nD', 'nXY', 'nX', 'nY', 'ifDef' )  

dataFile = fullfile( dataDir, 'dataAnomaly.mat' );
x = myData;
save( dataFile, '-v7.3', 'x', 'mu', 't', 'nD', 'nT', 'l2NormT', 'l2Norm' )
