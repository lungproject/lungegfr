close all
clear
load('./Alldata/sphmrn.mat');
sphtrainmrn = trainmrn;sphtestmrn = testmrn;
sphGroup2trains = Group2trains;sphGroup2tests = Group2tests;
load('./Alldata/hebeimrn.mat');
hebeitrainmrn = trainmrn;hebeitestmrn = testmrn;
hebeiGroup2trains = Group2trains;hebeiGroup2tests = Group2tests;
load('./Alldata/harbinmrn.mat');
harbintestmrn = testmrn;harbinGroup2tests = Group2s;
load('./Alldata/hlmmrns.mat');
hlmtestmrn = testmrn;hlmGroup2tests = Group2s;

load('./Alldata/ShanghaiClinical.mat')
sphclinical=ClinicalInfo;
load('./Alldata/HebeiClinical.mat')
hebeiclinical=ClinicalInfo;
load('./Alldata/HarbinClinical.mat')
harbinclinical=ClinicalInfo;
load('./Alldata/HLMClinicals.mat')
hlmclinical=ClinicalInfo;

sphtrainfeature = load('./Results/SPHpredicttrain.txt');
sphtestfeature = load('./Results/SPHpredicttest.txt');
hebeitrainfeature = load('./Results/Hebeipredicttrain.txt');
hebeitestfeature = load('./Results/Hebeipredicttest.txt');
hlmfeature = load('./Results/HLMpredicttests.txt');

harbinfeature = load('./Results/Harbinpredicttest.txt');
harbinIOfeature = load('./Results/Harbinpdlpredicttest.txt');

sphtrainact76=readNPY('./Results/activations76sphtrain.npy');
sphtestact76=readNPY('./Results/activations76sphtest.npy');

hebeitrainact76=readNPY('./Results/activations76hebeitrain.npy');
hebeitestact76=readNPY('./Results/activations76hebeitest.npy');



presphtrain = sphtrainfeature(:,2);
presphtest = sphtestfeature(:,2);
prehlmtest = hlmfeature(:,2);
preharbintest= harbinfeature(:,2);
prehebeitrain = hebeitrainfeature(:,2);
prehebeitest = hebeitestfeature (:,2);

hlmIOfeature = load('./Results/HLMPDLpredicttests.txt');
prehlmpdltest = hlmIOfeature(:,2);
preharbinpdltest= harbinIOfeature(:,2);
sphtraindata = unique(sphGroup2trains);
for i=1:length(sphtraindata)
    ind =  find(sphGroup2trains==sphtraindata(i)); 
     temp1 = presphtrain(ind,:);
     sphtrainpp(i,:)=mean(temp1); 
     
    temp1=sphtrainact76(ind,:,:,:);
     for j=1:256
          sphtrainact(i,j)=mean(temp1(:,:,j),'all');
    end
     
     [~,ind] = ismember(sphtrainmrn(ind(1)),sphclinical(:,1));
     sphtrainout(i,:) = [sphclinical(ind,1) sphtrainpp(i) double(sphtrainpp(i)>0.5) sphclinical(ind,2:end)];
 
 end
% 1.mrn2.dls 3sex 4age 5histology 6smoking 7stage 8SUVmax 9EGFR	10ALK 11ROS1 12EGFRClass
 
 sphtestdata = unique(sphGroup2tests);
 for i=1:length(sphtestdata)
     ind =  find(sphGroup2tests==sphtestdata(i)); 
     temp1 = presphtest(ind,:);
     sphtestpp(i,:)=mean(temp1); 
     
     temp1=sphtestact76(ind,:,:,:);
     for j=1:256
          sphtestact(i,j)=mean(temp1(:,:,j),'all');
     end
     
     
     [~,ind] = ismember(sphtestmrn(ind(1)),sphclinical(:,1));
     sphtestout(i,:) = [sphclinical(ind,1) sphtestpp(i) double(sphtestpp(i)>0.5) sphclinical(ind,2:end)];
 
 end
 
 
 hebeitraindata = unique(hebeiGroup2trains);
 for i=1:length(hebeitraindata)
     ind =  find(hebeiGroup2trains==hebeitraindata(i)); 
     temp1 = prehebeitrain(ind,:);
     hebeitrainpp(i,:)=mean(temp1); 
     
     temp1=hebeitrainact76(ind,:,:,:);
     for j=1:256
          hebeitrainact(i,j)=mean(temp1(:,:,j),'all');
     end
     
     
     [~,ind] = ismember(hebeitrainmrn(ind(1)),hebeiclinical(:,1));
     hebeitrainout(i,:) = [hebeiclinical(ind,1) hebeitrainpp(i) double(hebeitrainpp(i)>0.5) hebeiclinical(ind,2:end)];
 
 end
% 1.mrn	2.dls 3.sex 4.age	5.histology	6.smoking	7.stage 8.SUVmax	9.EGFR
 
 hebeitestdata = unique(hebeiGroup2tests);
 for i=1:length(hebeitestdata)
     ind =  find(hebeiGroup2tests==hebeitestdata(i)); 
     temp1 = prehebeitest(ind,:);
     hebeitestpp(i,:)=mean(temp1); 
     
     temp1=hebeitestact76(ind,:,:,:);
     for j=1:256
           hebeitestact(i,j)=mean(temp1(:,:,j),'all');
     end
     
         
     [~,ind] = ismember(hebeitestmrn(ind(1)),hebeiclinical(:,1));
     hebeitestout(i,:) = [hebeiclinical(ind,1) hebeitestpp(i) double(hebeitestpp(i)>0.5) hebeiclinical(ind,2:end)];
 
 end

harbintestdata = unique(harbinGroup2tests);
for i=1:length(harbintestdata)
    ind =  find(harbinGroup2tests==harbintestdata(i)); 
    temp1 = preharbinpdltest(ind,:);
    harbintestpp(i,:)=mean(temp1); 
    
  
    [~,ind] = ismember(harbintestmrn(ind(1)),harbinclinical(:,1));
    harbintestout(i,:) = [harbinclinical(ind,1) harbintestpp(i) double(harbintestpp(i)>0.5) harbinclinical(ind,2:end)];

end
%1mrn 2dls 3sex	4age 5histology	6SUV 7EGFR 8PRE 9response 10PFST 11PFS 12OST 13OS 14EGFRtype	15MPGSUV

hlmtestdata = unique(hlmGroup2tests);
for i=1:length(hlmtestdata)
    ind =  find(hlmGroup2tests==hlmtestdata(i)); 
    temp1 = prehlmtest(ind,:);
    hlmtestpp(i,:)=mean(temp1); 
    
    temp1 = prehlmpdltest(ind,:);
    hlmpdltestpp(i,:)=mean(temp1); 
    
 
    [~,ind] = ismember(hlmtestmrn(ind(1)),hlmclinical(:,1));
    if ind
     hlmtestout(i,:) = [hlmclinical(ind,1) hlmtestpp(i) hlmpdltestpp(i) double(hlmtestpp(i)>0.5) double(hlmpdltestpp(i)>0.5) hlmclinical(ind,2:22)];
    else
        hlmtestmrn(ind(1))
    end
    
   
end

% 1mrn,2dls,3SUVmax,4stage,5histology,6sex,7age,8bmi,9immuType,10smoke,11family,12copd,13ecog,14distmet,15pdltreta,
% 16pfst,17pfs,18bestgroup,19ost,20os,21group,22pdl,23EGFR,24ALK,25ROS1,26brain

trainall = [sphtrainout(:,1:10);hebeitrainout];
testall = [sphtestout(:,1:10);hebeitestout];


%%calculate the performance of the DLS
cutoff=0.5 ;
evetrain = EvaluationModel(trainall(:,2),trainall(:,10),1,cutoff);
evetest = EvaluationModel(testall(:,2),testall(:,10),1,cutoff);
eveharbin = EvaluationModel(harbintestout(:,2),harbintestout(:,10),1,cutoff);
evehlm = EvaluationModel(hlmtestout(:,2),hlmtestout(:,26),1,cutoff);

hlmtestout(:,[19 22])=hlmtestout(:,[19 22])/30;

% 




trainact = [sphtrainact;hebeitrainact];
testact = [sphtestact;hebeitestact];

%%deeply learned feature pattern

trainact = zscore(trainact,0,2);
testact = zscore(testact,0,2);


cg_s = clustergram(trainact, 'Cluster', 'All','Colormap','redbluecmap','RowPDist','correlation','ColumnPDist','correlation')
indc = cg_s.ColumnLabels;
indcs=[];
for i=1:length(indc)
    indcs = [indcs;str2num(indc{i})];
end

indr1 = cg_s.RowLabels;
indrs1=[];
for i=1:length(indr1)
    indrs1 = [indrs1;str2num(indr1{i})];
end

cg_s2 = clustergram(testact(:,indcs), 'Cluster', 'Column','Colormap','redbluecmap','RowPDist','correlation','ColumnPDist','correlation')
indr2 = cg_s2.RowLabels;
indrs2=[];
for i=1:length(indr2)
    indrs2 = [indrs2;str2num(indr2{i})];
end




