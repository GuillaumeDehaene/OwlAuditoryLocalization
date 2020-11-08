function [R1, R2, R3] = makeRCCSoundlocRespTrial_Multiple(F,tau,time,S1,S2,Fleft,Fright)
% compute predicted trial-by-trial response for filter-based models,
% several models at once
%
% Inputs:
%
% F: neural filters with preferred time lag = 0
%   size(F) = [#neurons #populations #timesteps]
% tau = vector of preferred time lags
% time = vector of time points at which responses are computed
% S1, S2: left and right ear inputs (#timepoints #trials)
% Fleft, Fright: time-shifted filters

%
% 2016.03 Ruben Coen-Cagli

if(~exist('Fleft'))
    Fleft=[];
    Fright=[];
end

[Kf, L] = size(F);
Kt1 = numel(tau);
Kt2 = numel(time);
K=(Kf*Kt1)*Kt2;

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

%%% several models at once
t1= (Fleft*S1);
t2= (Fright*S2);
Rbase = t1.*t2;

% 'LRQTSUM'
Rbase1 = Rbase;
Rbase1(Rbase1<0)=0;
R1 = Rbase1.^2;
R1 = reshape(R1,Kf,Kt2,Kt1,size(S1,2));
R1 = squeeze(sum(R1,2)) / Kt2;

%%% FISCHER2009 = Fischer Anderson Pena PLOS One 2009, sum over time points per neuron 
t11=t1./sqrt(1+t1.^2); % gain control
t22=t2./sqrt(1+t2.^2); % gain control
a = 70;
b = 10;
X = t11.*t22;
X(X<0)=0;
denom = 0.5*sum( t11.^2 + t22.^2 ,1) / Kt2;
X = X./repmat(denom.^.5,K,1);
X = reshape(X,Kf,Kt2,Kt1,size(S1,2));
X = squeeze(sum(X,2)) / Kt2;
R2 = 100./(1+exp(-a*X+b));
       
%%% LRDN2EXPTSUM = Linear-rectify-exp, sum over time points per neuron 
% After Cazettes et al 2016 
Rbase3 = Rbase;
Rbase3(Rbase3<0)=0; %rectify
Rbase3 = reshape(Rbase3,Kf,Kt2,Kt1,size(S1,2));
Rbase3 = squeeze(sum(Rbase3,2)) / Kt2; % sum over time points
R3 = exp(15*Rbase3);





    
