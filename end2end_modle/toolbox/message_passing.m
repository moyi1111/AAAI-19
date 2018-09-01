function [C,Energy]=message_passing_fast2(A,Aij,nn,pairterm,K,dumping_rate,tmax,small,threshold,beta,groundturth)

%topo = load('DATA/cora_n.mat');
%attr = load('DATA/cora_wd');
%attr = full(attr.cora_wd);
%attr = attr';
load DATA/cora_wd  
ALPHA = 1e-2;
BETA = 1e-2;
M = 1;    
SEED = 1;    
OUTPUT = 1;
J = 7;     


C=zeros(nn,1);%community

%Initialize phi theta and mu
N=1;
[phi, theta, mu] = csh(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
%banlance = norm(topo)/norm(attr);
%banlance = 1;
%theta = zeros(K,nn);
Belief=zeros(nn,K);%Initialize Belief
mss=rand(nn,nn,K);
mss=max(mss,small);
mss=log(mss);
minmss=min(mss,[],3);
minmss=repmat(minmss,[1,1,K]);
mss=mss-minmss;
mssnew=mss;

%iterative
t=1;
ST=true;
potential=zeros(nn,nn,K);
while (t<tmax)&&(ST==true)
   
               for k=1:K 
                   potential=repmat(pairterm,[1,1,K]);
                   potential(:,:,k)=-pairterm;
                   potential=beta*max(potential,small)+mss;
                   mama=max(potential,[],3);
                   smama=sum(mama)+theta(k,:);                 
                   N = t+10
                   [phi, theta, mu] = sBPtrain_down(cora_wd, J, Belief, ALPHA, BETA, SEED, OUTPUT,mu);
                   smatrix=repmat(smama,[nn,1]);
                   mmatrix=smatrix-mama;
                   for i=1:nn
                       mmatrix(i,i)=0;
                   end
                   mssnew(:,:,k)=mmatrix';
               end%END-(for k=1:K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%               
               %Ҳ���Գ����ȼ�ȥ���ֵ��ָ���ȥ��׼��
               minmss=min(mssnew,[],3);
               minmss=repmat(minmss,[1,1,K]);
               mssnew=mssnew-minmss;
               mssnew=dumping_rate*mssnew+(1-dumping_rate)*mss;

               Belief=zeros(nn,K);
                    for k=1:K 
                        potential=repmat(pairterm,[1,1,K]);
                        potential(:,:,k)=-pairterm;
                        potential=beta*max(potential,small)+mss;
                        mama=max(potential,[],3);
                            for i=1:nn
                                 mama(i,i)=0;
                            end
                        smama=sum(mama);
                        Belief(:,k)=smama';
                    end%END-( for k=1:K)
               
               
    t=t+1;
    mss_diff=abs(mssnew-mss);
    mss_dmax=max(max(max(mss_diff)));
    mss=mssnew;
    if (mss_dmax<threshold)
        ST=false;
    end
end%END-(while (t<tmax)&&(ST==true))


%Compute divide result?
Belief=zeros(nn,K);
     for k=1:K 
         potential=repmat(pairterm,[1,1,K]);
         potential(:,:,k)=-pairterm;
         potential=beta*max(potential,small)+mss;
         mama=max(potential,[],3);
         for i=1:nn
             mama(i,i)=0;
         end
         smama=sum(mama);
         Belief(:,k)=smama';
    end%END-( for k=1:K)
    
    [Mval,C]=max(Belief,[],2);


Ccol=repmat(C,[1,nn]);
Crow=repmat(C',[nn,1]);
Ctarget=Ccol==Crow;
Ctarget=(-1).^Ctarget;
for i=1:nn
    Ctarget(i,i)=0;
end
Menergy=beta*(Ctarget.*pairterm);
Energy=sum(sum(Menergy));    