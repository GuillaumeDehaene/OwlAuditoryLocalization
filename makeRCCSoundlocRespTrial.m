function [R, Fleft, Fright] = makeRCCSoundlocRespTrial(model,F,tau,time,S1,S2,Fleft,Fright)
% compute predicted trial-by-trial response for filter-based models
%
% Inputs:
%
% model (string):
%   L = Linear 
%   LR = Linear-rectified
%   LRP = Linear-rectified-Poisson
%   LQP = Linear-quadratic-Poisson
% F: neural filters with preferred time lag = 0
%   size(F) = [#neurons #populations #timesteps]
% tau = vector of preferred time lags
% time = vector of time points at which responses are computed
% S1, S2: left and right ear inputs (#timepoints #trials)
% Fleft, Fright: time-shifted filters

%
% 2015.10 Ruben Coen-Cagli

if(~exist('Fleft'))
    Fleft=[];
    Fright=[];
end

[Kf, L] = size(F);
Kt1 = numel(tau);
Kt2 = numel(time);
K=(Kf*Kt1)*Kt2;
R = zeros(K,1); % true response vector

if isempty(Fleft)
    Fleft = zeros(K,L);
    Fright = zeros(K,L);
    for t2=1:Kt2
        for t1=1:Kt1
            Fleft((1:Kf)+Kf*(t2-1+Kt2*(t1-1)),:) = circshift(F,[0 time(t2)]);
            Fright((1:Kf)+Kf*(t2-1+Kt2*(t1-1)),:) = circshift(F,[0 -tau(t1)+time(t2)]);
        end
    end
end


%%
switch model
    %%% 0 = return only the filter-product matrices
    case '0' %
    
    %%% L = Linear 
    case 'L' % 
        R = (Fleft*S1).*(Fright*S2);
    %%% LTSUM = Linear, sum over time points per neuron 
    case 'LTSUM' % 
        R = (Fleft*S1).*(Fright*S2); 
        R = reshape(R,Kf,Kt2,Kt1,size(S1,2));
        R = reshape(sum(R,2),Kf*Kt1,size(S1,2)) / Kt2;
    %%% LR = Linear-rectified
    case 'LR' %
        R = (Fleft*S1).*(Fright*S2);
        R(R<0)=0;
    %%% LRTSUM = Linear-rectify, sum over time points per neuron 
    case 'LRTSUM' % 
        R = (Fleft*S1).*(Fright*S2); 
        R(R<0)=0;
        R = reshape(R,Kf,Kt2,Kt1,size(S1,2));
        R = reshape(sum(R,2),Kf*Kt1,size(S1,2)) / Kt2;
    %%% LRDNTSUM = Linear-rectified with divisive normalization
    case 'LRDNTSUM' %
        R = (Fleft*S1).*(Fright*S2);
        R(R<0)=0;
        denom = 0.5*sum( (Fleft*S1).*(Fleft*S1) + (Fright*S2).*(Fright*S2) ,1) / Kt2;
        R = R./repmat(denom,K,1);
        R = reshape(R,Kf,Kt2,Kt1,size(S1,2));
        R = reshape(sum(R,2),Kf*Kt1,size(S1,2)) / Kt2;
    %%% LRQTSUM = Linear-rectify-square, sum over time points per neuron
    case 'LRQTSUM' % 
        R = (Fleft*S1).*(Fright*S2); 
        R(R<0)=0;
        R = R.^2;
        R = reshape(R,Kf,Kt2,Kt1,size(S1,2));
        R = reshape(sum(R,2),Kf*Kt1,size(S1,2)) / Kt2;
    %%% LRQDNTSUM = Linear-rectify-divnorm-square, sum over time points per neuron
    case 'LRQDNTSUM' % 
        R = (Fleft*S1).*(Fright*S2); 
        R(R<0)=0;
        
        denom = 0.5*sum( (Fleft*S1).*(Fleft*S1) + (Fright*S2).*(Fright*S2) ,1) / Kt2;
        R = R./repmat((denom+1).^.5,K,1);
        
        R = R.^2;
        R = reshape(R,Kf,Kt2,Kt1,size(S1,2));
        R = reshape(sum(R,2),Kf*Kt1,size(S1,2)) / Kt2;  
    %%% LDNEXPTSUM = Linear-divnorm, sum over time points per neuron, exp 
    case 'LDNEXPTSUM' % 
        t1= (Fleft*S1);
        t2= (Fright*S2);
        R = t1.*t2; 
        
        denom = 0.5*sum( t1.^2 + t2.^2 ,1) / Kt2;
        R = R./repmat(denom.^.5,K,1);
        R = reshape(R,Kf,Kt2,Kt1,size(S1,2));
        R = reshape(sum(R,2),Kf*Kt1,size(S1,2)) / Kt2;
        R = exp(R);
    %%% LRDNEXPTSUM = Linear-rectify-divnorm, sum over time points per neuron, exp 
    case 'LRDNEXPTSUM' % 
        t1= (Fleft*S1);
        t2= (Fright*S2);
        R = t1.*t2; 
        R(R<0)=0;
        
        denom = 0.5*sum( t1.^2 + t2.^2 ,1) / Kt2;
        R = R./repmat(denom.^.5,K,1);
        R = reshape(R,Kf,Kt2,Kt1,size(S1,2));
        R = reshape(sum(R,2),Kf*Kt1,size(S1,2)) / Kt2;
        R = exp(50*R);
    %%% FISCHER2009 = Fischer Anderson Pena PLOS One 2009, sum over time points per neuron 
    case 'FISCHER2009' % 
        t1= (Fleft*S1);
        t2= (Fright*S2);
        t1=t1./sqrt(1+t1.^2); % gain control
        t2=t2./sqrt(1+t2.^2); % gain control

        a = 70;
        b = 10;
        X = t1.*t2;
        X(X<0)=0;
        denom = 0.5*sum( t1.^2 + t2.^2 ,1) / Kt2;
        X = X./repmat(denom.^.5,K,1);
        X = reshape(X,Kf,Kt2,Kt1,size(S1,2));
        X = reshape(sum(X,2),Kf*Kt1,size(S1,2)) / Kt2;
        R = 100./(1+exp(-a*X+b));
        
    %%% LQ = Linear-quadratic
    case 'LQ' %
        R = (Fleft*S1).*(Fright*S2);
        R=R.^2;
    %%% LRQ = Linear-rectify-quadratic
    case 'LRQ' %
        R = (Fleft*S1).*(Fright*S2);
        R(R<0)=0;
        R=R.^2;
    %%% LX = Linear-exponential
    case 'LX' % 
        R = ((Fleft*S1).*(Fright*S2));
        R(R>100)=100;
        R=exp(R);
    %%% LRLOG = Linear-rectified, logarithm
    case 'LRLOG' %
        R = (Fleft*S1).*(Fright*S2);
        R(R<=0)=0;
        R=log(R+1);
    %%% LRDN = Linear-rectified with divisive normalization
    case 'LRDN' %
        R = (Fleft*S1).*(Fright*S2);
        R(R<0)=0;
        denom = 0.5*sum( (Fleft*S1).*(Fleft*S1) + (Fright*S2).*(Fright*S2) ,1) / Kt2;
        R = R./repmat(denom,K,1);
    
    %%% LRDNLOG = Performs full marginalization of nuisance variables
    %%% sigmaS and sigmaExt
    case 'LRDNLOG' %
        R = (Fleft*S1).*(Fright*S2);
        denom = 0.5*sum( (Fleft*S1).*(Fleft*S1) + (Fright*S2).*(Fright*S2) ,1) / Kt2;
        R = R./repmat(denom,K,1);
        tR2 = R.^2;
        R = -0.5*L*log(1-tR2) + log(normcdf((sqrt(L)/2)*R./sqrt(1+tR2)));
    %%% LRDNLOG = Performs full marginalization of nuisance variables
    %%% sigmaS and sigmaExt
    case 'LRTSUMDNLOG' %
        R = (Fleft*S1).*(Fright*S2);    
        R = reshape(R,Kf,Kt2,Kt1,size(S1,2));
        R = reshape(sum(R,2),Kf*Kt1,size(S1,2));    
        denom = 0.5*sum( (Fleft*S1).*(Fleft*S1) + (Fright*S2).*(Fright*S2) ,1) / Kt2;
        denom = repmat(denom,K,1);
        denom = reshape(denom,Kf,Kt2,Kt1,size(S1,2));
        denom = reshape(sum(denom,2),Kf*Kt1,size(S1,2));    
        R = R./denom;
        R = reshape(R,Kf,Kt1,size(S1,2));    
        R = squeeze(sum(R,1)) / Kt2;    
        tR2 = R.^2;
        R = -0.5*L*log(1-tR2) + log(normcdf((sqrt(L)/2)*R./sqrt(1+tR2)));
    
end
