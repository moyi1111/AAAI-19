
%Data Preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A=load('DATA/cora.dat');
Alabel=load('DATA/cora_rlabels_n.mat');
load DATA/cora_wd  

gt=Alabel.labels;
% A=[0,1;1,4;1,7;4,1;4,7;9,0;7,1];
% gt=[1,2,4,5,1,6,7,2,3,1];
he=find(A(:,1)==A(:,2));
A(he,:)=[];
elem=unique(A);
if elem(1)<1
   mv=1-elem(1);
   index=elem+mv;
else
    index=elem;
end
groundturth=gt(index);

nn=length(elem);
B=A;
for i=1:nn
    B(find(A==elem(i)))=i;
end
A=B;
clear i r c B;

B=[A(:,2),A(:,1)];
B=cat(1,A,B);
C=unique(B,'rows');

A=[C(:,2),C(:,1)];
clear B C;

LA=length(A);
VA=ones(LA,1);
Aij=sparse(A(:,1),A(:,2),VA,nn,nn);

lo=diff(A(:,2));
location=cat(1,1,lo);
location=cat(1,location,1);
identity=find(location);
degree=diff(identity);
doublem=sum(degree);

%Initialize the pairwise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
small=1e-20;
he=degree*degree';  
he2=he/max(doublem,small);  
pairterm=he2-Aij;  
for i=1:nn
    pairterm(i,i)=0;
end



%Parameter Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_run=1;
K=length(unique(groundturth));
dumping_rate=1;%from 0-1
tmax=20;%The number of iterations?
small=1e-20;%In the case where the value is 0?
threshold=1e-20;%Determine to be the minimum threshold of convergence

beta=1000;% Parameter of the MRF part

%another way to get the beta
%avg_d=doublem/nn;
%temp=(K/(sqrt(avg_d)-1))+1;
%beta=log(max(temp,small));


% paramenter of the LDA part
ALPHA = 0.01 %1e-2;
BETA = 0.01 %1e-2;
M = 1;     
SEED = 1;    
OUTPUT = 1;
J = 7;     %the value of J is the same of K in MRF


%message passing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Energy=zeros(num_run,1);
results=zeros(num_run,nn);
for i=1:num_run
    [results(i,:),Energy(i)]=message_passing(A,Aij,nn,pairterm,K,dumping_rate,tmax,small,threshold,beta,groundturth);
end           
              
               
%Compute AC NMI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[MEnergy,IEnergy]=max(Energy);
Eindex=find(Energy==MEnergy);
maxln=length(Eindex);
rn=floor(rand(1)*maxln)+1; 
TEnergy=Eindex(rn);

partition=results(TEnergy,:);
NMI=compute_NMI(groundturth,partition);
ac=compute_AC(groundturth,partition);
save('cora_v1_result','partition');
fprintf('cora v1 NMI=%f\n',NMI);
fprintf('cora v1 ac=%f\n',ac);