% This file computes some experiments and figures of Dehaene, Coen-Cagli,
% Pouget 2019
% This file computes KL divergence and R-square figures for the Cazettes IC
% model
%
% IMPORTANT: to run this file, please add to path (select both -> right click -> add to path -> selected folders and subfolders)
% the following two folders
%     - maxent
%     - lbfgs

clear all
FS=20;  % Fontsize for figures
targetVar = true    % If true, the uncertainty is defined as the posterior variance
                    % If false, the uncertainty is defined as the posterior
                    % entropy instead (as in the supplementary information)

load('OracleLlAndInfo_ProbTrainLRTSUM_3Batch2000_T01_BC1','NBC','Kf','Kt1','N','Nbatch')

modelnlin = 'LRTSUM';

% We pre-allocate key quantities for memory storage

KLneulin = zeros(NBC,1);
KLneunlin = KLneulin;
seKLneulin = zeros(NBC,2);
seKLneunlin = seKLneulin;

nboot=100;

Rsquare_Rlin = zeros(NBC,nboot);
Rsquare_Rnlin = Rsquare_Rlin;
Rsquare_RnlinProd = Rsquare_Rlin;
Rsquare_gRnlin = Rsquare_Rlin;
Rsquare_widthRnlin = Rsquare_Rlin;
Rsquare_gwidthRnlin = Rsquare_Rlin;
Rsquare_decodeRlin = Rsquare_Rlin;
Rsquare_decodeRnlin = Rsquare_Rlin;

%%%%% 
% Main computational loop
% For every BC level, we compute a bunch of things

% oracle situation: we condition on a specific BC level
for indNBC=1:NBC
    s = indNBC;
    load(strcat('OracleLlAndInfo_ProbTrain',modelnlin,'_',num2str(N/Nbatch),'Batch',num2str(Nbatch),'_T01_BC',num2str(s)));
    s = indNBC;

    % load the Cazettes data
    Rnlin = Rnlin3;
    BAllnlin = BAllnlin3;
    
    itdSubset = 5:7;   % subselect the ITDs
    KitdSubset = size(itdSubset,2);
    
    LL = LL(:,:,itdSubset,:); %*** loglikelihood for just one true ITD
    LLneulin=LL;            % pre-allocate memory
    LLneunlin=LL;           % pre-allocate memory
    Rlin  = Rlin (:,:,itdSubset,:);
    Rnlin = Rnlin(:,:,itdSubset,:);
    
    % approximate decoding of the neuronal activity
    % the weights are stored into the BAlllin variable and have been
    % computed earlier
    X = reshape(Rlin,[Kf*Kt1 KitdSubset*N])';    % removed inddt2
    LLneulin(s,:,:,:) = reshape( energy(BAlllin,X)', [Kt1, KitdSubset, N]);
    X = reshape(Rnlin,[Kf*Kt1 KitdSubset*N])';    % removed inddt2
    LLneunlin(s,:,:,:) = reshape( energy(BAllnlin,X)', [Kt1, KitdSubset, N]);
    
    % extract test set
    % train set was used to construct the decoding weights earlier
    LL = squeeze(LL(s,:,:,indtest));
    LLneulin  = squeeze(LLneulin (s,:,:,indtest));
    LLneunlin = squeeze(LLneunlin(s,:,:,indtest));
    Rlin  = Rlin(:,:,:,indtest);
    Rnlin = Rnlin(:,:,:,indtest);

    
    % select all trials
    indtri = 1:size(indtest,2);
    NT = size(indtri,2);
    
    %%% compute full posterior (w unif prior), per trial
    tmp = nanmax(LL(:,indtri),[],1); % avoid infinities in the likelihood function
    LL_norm = bsxfun(@minus,LL(:,indtri),tmp);
    tmp = nanmax(LLneulin(:,indtri),[],1); % avoid infinities in the likelihood function
    LLneulin_norm = bsxfun(@minus,LLneulin(:,indtri),tmp);
    tmp = nanmax(LLneunlin(:,indtri),[],1); % avoid infinities in the likelihood function
    LLneunlin_norm = bsxfun(@minus,LLneunlin(:,indtri),tmp);
    
    % normalized posterior distributions
    L_norm = bsxfun(@rdivide,exp(LL_norm),nansum(exp(LL_norm),1)); % compute likelihood
    Lneulin_norm = bsxfun(@rdivide,exp(LLneulin_norm),nansum(exp(LLneulin_norm),1));
    Lneunlin_norm = bsxfun(@rdivide,exp(LLneunlin_norm),nansum(exp(LLneunlin_norm),1));
    
    %%% compute KL divergence between true and reconstructed posterior
    NDT2 = size(L_norm,1);
    doeps=1;
    tDenom = diag(L_norm'*(log(L_norm+eps*doeps)-log(ones(NDT2,NT)/NDT2)));
    tKLneulin = diag(L_norm'*(log(L_norm+eps*doeps)-log(Lneulin_norm+eps*doeps)))   ./ tDenom;
    tKLneunlin = diag(L_norm'*(log(L_norm+eps*doeps)-log(Lneunlin_norm+eps*doeps)))  ./ tDenom;
    KLneulin(s) = nanmedian(tKLneulin);
    KLneunlin(s) = nanmedian(tKLneunlin);
    nboot1=1000;
    seKLneulin(s,:) = bootci(nboot1,{@nanmedian,tKLneulin},'type','bca');
    seKLneunlin(s,:) = bootci(nboot1,{@nanmedian,tKLneunlin},'type','bca');

    %%% compute posterior width
    x=deltaTrue(1:end);
    muL = x*L_norm;
    varL = sum(L_norm.*(repmat(x',1,size(L_norm,2))-repmat(muL,numel(x),1)).^2);
    ivarL = 1./varL;
    muL = x*Lneulin_norm;
    varL = sum(Lneulin_norm.*(repmat(x',1,size(Lneulin_norm,2))-repmat(muL,numel(x),1)).^2);
    ivarLneulin = 1./varL;
    muL = x*Lneunlin_norm;
    varL = sum(Lneunlin_norm.*(repmat(x',1,size(Lneunlin_norm,2))-repmat(muL,numel(x),1)).^2);
    ivarLneunlin = 1./varL;
    
    %%% compute entropy
    entropyL        = - diag( L_norm'         * log(L_norm) );        % the diagonal of U^T U corresponds to a sum
    entropyLneulin  = - diag( Lneulin_norm'   * log(Lneulin_norm) );  
    entropyLneunlin = - diag( Lneunlin_norm'  * log(Lneunlin_norm) );
    
    %%% extract features from population activity (width, gain) to correlate with uncertainty; also, single neuron tuning curve width and gain
    tRlin   = reshape( Rlin (:,:,:,indtri), [Kf, Kt1, KitdSubset * NT] );    % removed inddt2
    tRnlin  = reshape( Rnlin(:,:,:,indtri), [Kf, Kt1, KitdSubset * NT] );    % removed inddt2
    muRlin      = zeros(Kf,KitdSubset*NT);
    widthRlin   = zeros(Kf,KitdSubset*NT);
    muRnlin     = zeros(Kf,KitdSubset*NT);
    widthRnlin  = zeros(Kf,KitdSubset*NT);
    gwidthRnlin = zeros(Kf,KitdSubset*NT);
    gRnlin      = zeros(Kf,KitdSubset*NT);
    for f=1:Kf
        %%% population, per trial
        tmp = squeeze(tRlin(f,:,:));
        tmp = bsxfun(@minus,tmp,nanmin(tmp,[],1));
        tmpnorm = bsxfun(@rdivide,tmp,nansum(tmp,1));
        muRlin(f,:) = x*tmpnorm;
        widthRlin(f,:) = sum(tmpnorm.*bsxfun(@minus,x',muRlin(f,:)).^2);
        tmp = squeeze(tRnlin(f,:,:));
        gRnlin(f,:) = nansum(tmp);
        tmpnorm = bsxfun(@rdivide,tmp,nansum(tmp,1));
        muRnlin(f,:) = x*tmpnorm;
        widthRnlin(f,:) = sum(tmpnorm.*bsxfun(@minus,x',muRnlin(f,:)).^2);        
        gwidthRnlin(f,:) = gRnlin(f,:)./widthRnlin(f,:);
    end
    
    %%% population, per trial
    tmp = squeeze(sum(tRlin));
    tmp = bsxfun(@minus,tmp,nanmin(tmp,[],1));
    tmpnorm = bsxfun(@rdivide,tmp,nansum(tmp,1));
    mutmp = x*tmpnorm;
    widthRlinSum = sum(tmpnorm.*bsxfun(@minus,x',mutmp).^2);
    tmp = squeeze(sum(tRnlin));
    gRnlinSum = sum(tmp);
    tmp = bsxfun(@minus,tmp,nanmin(tmp,[],1));
    tmpnorm = bsxfun(@rdivide,tmp,nansum(tmp,1));
    mutmp = x*tmpnorm;
    widthRnlinSum = sum(tmpnorm.*bsxfun(@minus,x',mutmp).^2);
 
    %%% Define the target of the regression
    if targetVar
        %%% Target log of certainty 
        ivarL = log(ivarL); 
        ivarLneulin = log(ivarLneulin); 
        ivarLneunlin = log(ivarLneunlin); 
    else
        %%% Target the log entropy
        ivarL        = log(entropyL)';
        ivarLneulin  = log(entropyLneulin)';
        ivarLneunlin = log(entropyLneunlin)';
    end

    %%% take log of inverse of population width
    widthRlin = log(1./widthRlin);
    widthRnlin = log(1./widthRnlin);
    
    %%% take log of population gain
    gRnlin = log(gRnlin);
    gwidthRnlin = log(gwidthRnlin);
    
    %%% compute R-square of true certainty vs reconstruction from poulation features (ENCODING)
    NTT = NT;
    NTtr = round(4*NTT/5);
    NTte = numel(NTtr+1:NTT);
    for k=1:nboot
        indperm = randperm(NT);
        indtr = indperm(1:NTtr);
        indte = indperm((1+NTtr):end);
        
        tmp=reshape(tRlin(:,:,indtr),Kf*Kt1,NTtr); %*** LINEAR 
        tmp1=reshape(tRlin(:,:,indte),Kf*Kt1,NTte);
        B = regress(ivarL(indtr)',[tmp' ones(NTtr,1)]);
        Rsquare_Rlin(s,k) = (1-sum((ivarL(indte)'-[tmp1' ones(NTte,1)]*B).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));
        
        tmp=reshape(tRnlin(:,:,indtr),Kf*Kt1,NTtr); %*** LRTSUM
        tmp1=reshape(tRnlin(:,:,indte),Kf*Kt1,NTte);
        B = regress(ivarL(indtr)',[tmp' ones(NTtr,1)]);
        Rsquare_Rnlin(s,k) = (1-sum((ivarL(indte)'-[tmp1' ones(NTte,1)]*B).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));
        
        tmp1=reshape(tRnlin(:,:,indtr),Kf*Kt1,NTtr); %*** LRTSUM, product of activities
        Kneu = Kf*Kt1;
        Kneusq = (Kneu^2+Kneu)/2;
        tmp = NaN(Kneusq,NTtr);
        for nt = 1:NTtr
            tmpmat = tmp1(:,nt)*tmp1(:,nt)';
            tmp(:,nt) = tmpmat(logical(tril(ones(Kneu))));
        end     
        tmp1=reshape(tRnlin(:,:,indte),Kf*Kt1,NTte);
        testRnlinProd = NaN(Kneusq,NTte);
        for nt = 1:NTte
            tmpmat = tmp1(:,nt)*tmp1(:,nt)';
            testRnlinProd(:,nt) = tmpmat(logical(tril(ones(Kneu))));
        end       
        if NTtr<=Kneusq
            indsub = randperm(Kneusq);
            tmp = tmp(indsub(1:round(NTtr/2)),:);
            testRnlinProd = testRnlinProd(indsub(1:round(NTtr/2)),:);
        end
        B = regress(ivarL(indtr)',[tmp' ones(NTtr,1)]);
        Rsquare_RnlinProd(s,k) = (1-sum((ivarL(indte)'-[testRnlinProd' ones(NTte,1)]*B).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));
        
        tmp=gRnlin(:,indtr);
        tmp1=gRnlin(:,indte);
        B = regress(ivarL(indtr)',[tmp' ones(NTtr,1)]);
        Rsquare_gRnlin(s,k) = (1-sum((ivarL(indte)'-[tmp1' ones(NTte,1)]*B).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));
        tmp=widthRnlin(:,indtr);
        tmp1=widthRnlin(:,indte);
        B = regress(ivarL(indtr)',[tmp' ones(NTtr,1)]);
        Rsquare_widthRnlin(s,k) = (1-sum((ivarL(indte)'-[tmp1' ones(NTte,1)]*B).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));
        tmp=gwidthRnlin(:,indtr);
        tmp1=gwidthRnlin(:,indte);
        B = regress(ivarL(indtr)',[tmp' ones(NTtr,1)]);
        Rsquare_gwidthRnlin(s,k) = (1-sum((ivarL(indte)'-[tmp1' ones(NTte,1)]*B).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));

        if 0 %%% compute R-square of true certainty vs reconstruction from poulation activtiy (DECODING) ***with*** linear fit
            tmp=ivarLneulin(indtr);
            tmp1=ivarLneulin(indte);
            B = regress(ivarL(indtr)',[tmp' ones(NTtr,1)]);
            Rsquare_decodeRlin(s,k) = (1-sum((ivarL(indte)'-[tmp1' ones(NTte,1)]*B).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));
            tmp=ivarLneunlin(indtr);
            tmp1=ivarLneunlin(indte);
            B = regress(ivarL(indtr)',[tmp' ones(NTtr,1)]);
            Rsquare_decodeRnlin(s,k) = (1-sum((ivarL(indte)'-[tmp1' ones(NTte,1)]*B).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));
        else %%% compute R-square of true certainty vs reconstruction from poulation activtiy (DECODING) ***without*** linear fit
            Rsquare_decodeRlin(s,k) = (1-sum((ivarL(indte)-ivarLneulin(indte)).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));
            Rsquare_decodeRnlin(s,k) = (1-sum((ivarL(indte)-ivarLneunlin(indte)).^2)/sum((ivarL(indte)-mean(ivarL(indte))).^2));
        end
    end
end    


%%%%% 
% All computations are complete: now we plot the figures

%myerrorbar = @(x,y,z,a,b) errorbar(x,y,y - z(:,1),z(:,2)-y);

%%% plot KL vs BC
figure; 
hold on; axis square;
myerrorbar(BinCorLabel',KLneunlin,seKLneunlin,[.6 .6 .6],1);
plot(BinCorLabel',KLneunlin,'-k');
xlabel('BC','FontSize',FS)
ylabel('KL','FontSize',FS)
set(gca,'YLim',[0 .15],'YTick',[0:.01:0.1],'YTickLabel',100*[0:.01:0.1])

%%% plot Rsquare vs BC
figure; 
alpha=0.01;
nboot2=1000;
shade=1;
hold on; axis square;
% myerrorbar(BinCorLabel,nanmedian(Rsquare_decodeRlin'),bootci(nboot2,{@nanmedian,Rsquare_decodeRlin'},'alpha',alpha)','k',shade);
myerrorbar(BinCorLabel',nanmedian(Rsquare_RnlinProd')',   bootci(nboot2,{@nanmedian,Rsquare_RnlinProd'},      'alpha',alpha)','m',shade);
myerrorbar(BinCorLabel',nanmedian(Rsquare_gwidthRnlin')', bootci(nboot2,{@nanmedian,Rsquare_gwidthRnlin'},    'alpha',alpha)','y',shade);
myerrorbar(BinCorLabel',nanmedian(Rsquare_Rnlin')',       bootci(nboot2,{@nanmedian,Rsquare_Rnlin'},          'alpha',alpha)','b',shade);
myerrorbar(BinCorLabel',nanmedian(Rsquare_gRnlin')',      bootci(nboot2,{@nanmedian,Rsquare_gRnlin'},         'alpha',alpha)','r',shade);
myerrorbar(BinCorLabel',nanmedian(Rsquare_widthRnlin')',  bootci(nboot2,{@nanmedian,Rsquare_widthRnlin'},     'alpha',alpha)','g',shade);
myerrorbar(BinCorLabel',nanmedian(Rsquare_decodeRnlin')', bootci(nboot2,{@nanmedian,Rsquare_decodeRnlin'},    'alpha',alpha)','k',shade);
legend({'Generalized Gain + Product (??)','Gain / Width Ratio (??)','Generalized Gain (??)','Gain (??)','Width (??)','Decoding (??)',})
xlabel('BC','FontSize',FS)
ylabel('Rsquare(certainty)')
set(gca,'YLim',[0 1],'YTick',[0:.2:1])
