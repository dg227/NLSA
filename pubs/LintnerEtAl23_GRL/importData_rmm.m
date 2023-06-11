function importData_rmm(P)
% Read RMM data in format appropriate for NLSA code. 
%
% P: Input parameter structure. 
%
% Modified 2023/06/07

    names = {
        'pth' 
        'dataset'
        'file'
        'tFormat'
        'tStart'
        'tLim'
        'xLim'
        'yLim'
        'var'
        };

    args = namedargs2cell(filterField(P, names));
    importData(args{:}); 
end


function importData(P)
    arguments
        P.pth     (1, :) {mustBeText} = './data/raw'
        P.dataset (1, :) {mustBeText} = 'LGPS23'
        P.file    (1, :) {mustBeText} = 'RMM_data.mat'
        P.tFormat (1, :) {mustBeTextScalar} = 'yyyymmdd'
        P.tStart  (1, :) {mustBeTextScalar} = '19980101'
        P.tLim    (1, 2) {mustBeText} = {'19980101' '20191230'}
        P.xLim    (1, 2) {mustBeNumeric} = [40 290]
        P.yLim    (1, 2) {mustBeNumeric} = [-15 15]
        P.var     (1, :) {mustBeTextScalar} = 'RMM'
    end

    tStr = strjoin_e(P.tLim, '-');
    xyStr = sprintf('x%i-%i_y%i-%i', P.xLim(1), P.xLim(2), P.yLim(1), P.yLim(2));

    inDir = fullfile(P.pth, P.dataset);
    outDir = fullfile(P.pth, P.dataset, P.var, [xyStr '_' tStr]);

    X = load(fullfile(inDir, P.file), P.var);
    x = X.(P.var);
    nD = size(x, 1);
    startNum = datenum(P.tStart, P.tFormat);
    limNum   = datenum(P.tLim, P.tFormat); 
    idxTLim  = limNum - startNum + 1;
    x = x(:, idxTLim(1) : idxTLim(2));

    if ~isdir(outDir)
        mkdir(outDir)
    end
    save(fullfile(outDir, 'dataX.mat'), 'x') 
    save(fullfile(outDir, 'dataGrid.mat'), 'nD') 
end
