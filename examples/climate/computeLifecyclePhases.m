function [ selectind, angles, avbindex ] = computeLifecyclePhases( ...
    phi, refI, nPhase, num, skipfraction )
% Compute lifecycle phase based on eigenfunction data phi. refI is a reference
% index used to align the phases such that the first phase corresponds to the
% largest phase-average nino value. 

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
    anglemin=min([abs(angle(complex(phi(:,1),phi(:,2)))-theta) abs(angle(complex(phi(:,1),phi(:,2)))+2*pi-theta) abs(angle(complex(phi(:,1),phi(:,2)))-2*pi-theta)],[],2);
    ind=find(anglemin<pi*wedgefraction);
             
    [y,sorti]=sort(phi(ind,1).^2+phi(ind,2).^2,'descend');
    nSort = min( num, numel( sorti ) );
    rotind{count}=ind(sorti(1:nSort));
    avbindex(count)=mean(refI(rotind{count}));
end
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
starttheta=-pi+(i-1)*skipfraction*2*pi;
angles=mod(starttheta+pi:wedgefraction*2*pi:starttheta+3*pi-wedgefraction*pi,2*pi)-pi;
count=0;
selectind=cell(1/wedgefraction,1);
for theta=angles,
    count=count+1;
    %ind=find(min(abs(angle(complex(phi(:,1),phi(:,2)))-theta),abs(angle(complex(phi(:,1),phi(:,2)))+2*pi-theta))<pi*wedgefraction);
    anglemin=min([abs(angle(complex(phi(:,1),phi(:,2)))-theta) abs(angle(complex(phi(:,1),phi(:,2)))+2*pi-theta) abs(angle(complex(phi(:,1),phi(:,2)))-2*pi-theta)],[],2);
    ind=find(anglemin<pi*wedgefraction);
    [y,sorti]=sort(phi(ind,1).^2+phi(ind,2).^2,'descend');
    nSort = min( num, numel( sorti ) );
    selectind{count}=ind(sorti(1:nSort));
    avbindex(count)=mean(refI(selectind{count}));
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
