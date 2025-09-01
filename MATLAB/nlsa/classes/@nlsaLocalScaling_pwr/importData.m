function D = importData(obj, dat, iC, iB, iR)
%% IMPORTDATA  Get scaling data from an nlsaDistanceData_scl object
%
% Modified 2022/11/06

dat = getSclComponent(dat);
nE  = getEmbeddingWindow(dat);

if iC > 1 && size(dat, 1) == 1 
    iC = 1;
end

switch getMode(obj)
    case 'explicit'
        outFormat = 'evector';
    case 'implicit'
        outFormat = 'overlap';
end
D.p = getExponent(obj, iC);
D.q = getData(dat(iC, iR), iB, outFormat);

