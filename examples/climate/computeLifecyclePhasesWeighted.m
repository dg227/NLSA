function [ selectind, angles, avbindex, weight ] = computeLifecyclePhasesWeighted( ...
    phi, refI, nPhase, num, skipfraction )
% Compute lifecycle phase based on eigenfunction data phi. refI is a reference
% index used to align the phases such that the first phase corresponds to the
% largest phase-average nino value.

% 'weight' is a cell array with nPhase cells (just like selectind), and where
% each cell is a vector of the same length as phi, with entries
% representing the weight of each data point.  The sum of each of these
% vectors is 1 (they are probability vectors).

% default input arguments
if nargin < 6
    ifPlot = false;
end
if nargin < 5
    skipfraction = 1 / 120;
end

wedgefraction = 1 / nPhase;

rotind=cell(1/skipfraction,1);
count=0;
for theta=-pi:2*pi*skipfraction:pi-2*pi*skipfraction,
    count=count+1;
    %ind=find(min(abs(angle(complex(phi(:,1),phi(:,2)))-theta),abs(angle(complex(phi(:,1),phi(:,2)))+2*pi-theta))<pi*wedgefraction);
    %find the angle between each point in phi-embedded space and theta
    anglemin=min([abs(angle(complex(phi(:,1),phi(:,2)))-theta) abs(angle(complex(phi(:,1),phi(:,2)))+2*pi-theta) abs(angle(complex(phi(:,1),phi(:,2)))-2*pi-theta)],[],2);
    %find the indices of those embedded points that lie within a wedge of
    %theta
    ind=find(anglemin<pi*wedgefraction);
    
    %sort the magnitude of these embedded points in the wedge
    [y,sorti]=sort(phi(ind,1).^2+phi(ind,2).^2,'descend');
    nSort = min( num, numel( sorti ) );
    %put the largest magnitude indices in a cell
    rotind{count}=ind(sorti(1:nSort));
    %compute the average Nino34 index for the wedge.
    avbindex(count)=mean(refI(rotind{count}));
end
%select the wedge with the maximum Nino34 value
[y,i]=max(avbindex);

if ifPlot
    figure
    plot(phi(:,1),phi(:,2),'.')
    hold
    plot(phi(rotind{i},1),phi(rotind{i},2),'o')
    axis equal
    axis tight
    xlabel('phi_2')
    ylabel('phi_3')
    title('Strongest nino34index wedge from IndoPacific vector embedding')
end

%% having fixed the phase, compute the wedges

clear avbindex
%start at the wedge with maximal Nino34 value
starttheta=-pi+(i-1)*skipfraction*2*pi;
%step in jumps of 2*pi/nPhases
angles=mod(starttheta+pi:wedgefraction*2*pi:starttheta+3*pi-wedgefraction*pi,2*pi)-pi;
count=0;
selectind=cell(1/wedgefraction,1);
for theta=angles,
    count=count+1;
    weight{count}=zeros(size(phi,1),1);
    %find the angle between each point in phi-embedded space and theta
    anglemin=min([abs(angle(complex(phi(:,1),phi(:,2)))-theta) abs(angle(complex(phi(:,1),phi(:,2)))+2*pi-theta) abs(angle(complex(phi(:,1),phi(:,2)))-2*pi-theta)],[],2);
    %find the indices of those embedded points that lie within a wedge of
    %theta
    ind=find(anglemin<pi*wedgefraction);
    %sort the magnitude of these embedded points in the wedge
    [y,sorti]=sort(phi(ind,1).^2+phi(ind,2).^2,'descend');
    nSort = min( num, numel( sorti ) );
    %put the largest magnitude indices in a cell
    selectind{count}=ind(sorti(1:nSort));
    weight{count}(selectind{count})=(1/2)/length(selectind{count});
    %compute the average Nino34 index for the wedge.
    avbindex(count)=mean(refI(selectind{count}));
end

%% having found all the phases, assign weights to data points

%start at the wedge with maximal Nino34 value
starttheta=-pi+(i-1)*skipfraction*2*pi;
%step in jumps of 2*pi/nPhases
angles=mod(starttheta+pi:wedgefraction*2*pi:starttheta+3*pi-wedgefraction*pi,2*pi)-pi;
count=0;
for theta=angles,
    count=count+1;
    %countup is the incremented count (to access one wedge 'ahead'),
    %similarly countdown
    countup=mod((count-1)+1,nPhase)+1;
    countdown=mod((count-1)-1,nPhase)+1;
    %thetaup is the 'righthand' edge of the current central wedge,
    %similarly thetadown is the 'lefthand' edge.
    thetaup=theta+wedgefraction*pi;
    thetadown=theta-wedgefraction*pi;
    %angleup is the angle that all selected points in the one step advanced wedge make with the leading edge of the central wedge, similarly angledown
    angleup=min([abs(angle(complex(phi(selectind{countup},1),phi(selectind{countup},2)))-thetaup) abs(angle(complex(phi(selectind{countup},1),phi(selectind{countup},2)))+2*pi-thetaup) abs(angle(complex(phi(selectind{countup},1),phi(selectind{countup},2)))-2*pi-thetaup)],[],2);
    angledown=min([abs(angle(complex(phi(selectind{countdown},1),phi(selectind{countdown},2)))-thetadown) abs(angle(complex(phi(selectind{countdown},1),phi(selectind{countdown},2)))+2*pi-thetadown) abs(angle(complex(phi(selectind{countdown},1),phi(selectind{countdown},2)))-2*pi-thetadown)],[],2);
    %apply weights so that the total weight of the points in the advanced
    %wedge is 1/4, similarly for the previous wedge.
    weight{count}(selectind{countup})=1-angleup/(wedgefraction*2*pi);
    weight{count}(selectind{countup})=weight{count}(selectind{countup})/sum(weight{count}(selectind{countup}))*(1/4);
    weight{count}(selectind{countdown})=1-angledown/(wedgefraction*2*pi);
    weight{count}(selectind{countdown})=weight{count}(selectind{countdown})/sum(weight{count}(selectind{countdown}))*(1/4);
end

if ifPlot
    figure
    plot(phi(:,1),phi(:,2),'.')
    hold on
    for i=1:nPhase,
        plot(phi(selectind{i},1),phi(selectind{i},2),'o','markersize',10)
        axis equal
        axis tight
        xlabel('phi_1')
        ylabel('phi_2')
        title('ENSO lifecycle from IndoPacific vector embedding')
    end
end

if ifPlot
    figure
    for i=1:8,
        subplot(2,4,i)
        scatter(phi(:,1),phi(:,2),5,weight{i});colormap jet;colorbar;axis tight;axis equal; title(['Wedge ' num2str(i)])
    end
end

