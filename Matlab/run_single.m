% perform a single GMLVQ training process
% optional visualization of training curves and gmlvq system

function [gmlvq_system]= ...
   run_single(fvec,lbl,totalsteps,plbl)    
 
% gmlvq with global matrix only, square Omega, potentially diagonal
% batch gradient descent with step size control
% following a modified Papari procedure (unpublished) 
% perform training based using fvec, lbl
% evaluate performance measures after each step
% with respect to training set and validation set

% input arguments    
% fvec, lbl              training data, feature vectors and labels
% plbl: labels assigned to prototypes, also specifies number per class
% e.g. plbl=[1,1,2,2,3] for 5 prototypes with labels 1,2,3  
% totalsteps: number of batch gradient steps to be performed

if (nargin<4||isempty(plbl)); plbl=[1:length(unique(lbl))]; 
display('default: one prototype per class'); 
end;
display('prototype configuration'); plbl

if (nargin<3||isempty(totalsteps)); totalsteps=10; 
display('default number of training steps'); end;
% default: 10 gradient steps only 
display('number of training steps'); totalsteps

% general algorithm settings and parameters of the Papari procedure
[showplots,doztr,mode,rndinit, etam0, etap0, mu, decfac, incfac, ncop] =...
                                        set_parameters(fvec); 
etam=etam0;  % initial step size matrix
etap=etap0;  % intitial step size prototypes
                                    
% showplots (0 or 1): plot learning curves etc? 
% doztr (0 or 1): perform z-score transformation based on training set
% mode 
  % 0 for matrix without null-space correction
  % 1 for matrix with null-space correction
  % 2 for diagonal matrix (GRLVQ)                    DISCOURAGED
  % 3 for GLVQ with Euclidean distance (equivalent)
% rndinit
  % 0 for initialization of relevances as identity matrix 
  % 1 for randomized initialization of relevance matrix 
% etam:    step size of matrix updates
% etap:    step size of prototype updates
% mu  :    control parameter of penalty term for singular Lambda
% decfac:  factor for decrease of step sizes for oscillatory behavior
% incfac:  factor for increase of step sizes for smooth behavior
% ncop:    number of copies in Papari procedure

% check for consistency and output error messages
% transpose lbl if necessary

[lbl]=check_arguments(plbl,lbl,fvec,ncop,totalsteps); 

close all;   % close all figures

% reproducible random numbers
 rng('default'); 
 rngseed=291024;
 rng(rngseed);

nfv = size(fvec,1);          % number of feature vectors in training set
ndim = size(fvec,2);         % dimension of feature vectors
ncls = length(unique(lbl));  % number of classes 
nprots = length(plbl);       % total number of prototypes

% comment out for cost function
% te=zeros(totalsteps+1,1);      % define total error
% cf=te; auc=te;               % define cost function and AUC(ROC)
% cw=zeros(totalsteps+1,ncls);   % define class-wise errors

      
      mf=zeros(1,ndim);      % initialize feature means
      st=ones(1,ndim);       % and standard deviations
if doztr==1;
      [fvec,mf,st] = do_zscore(fvec);  % perform z-score transformation
else  [~,mf,st]= do_zscore(fvec);      % evaluate but don't apply 
end; 

% initialize prototypes and omega 
  [proti,omi] =  set_initial(fvec,lbl,plbl,mode,rndinit);
  prot=proti;  om =omi;   % initial values
  
% copies of prototypes and omegas stored in protcop and omcop
% for the adaptive step size procedure 
  protcop = zeros(ncop,size(prot,1),size(prot,2));
  omcop   = zeros(ncop,size(om,1) , size(om,2) );

  % calculate initial values for learning curves
  % [costf,~,marg,score] = compute_costs(fvec,lbl,prot,plbl,om,mu);
  %     te(1) = sum(marg>0)/nfv;
  %     cf(1) = costf;

  % perform the first ncop steps of gradient descent
  for inistep=1: ncop;
      % actual batch gradient step
      
      [prot,om]= do_batchstep(fvec,lbl,prot,plbl,om,etap,etam,mu,mode); 
      protcop(inistep,:,:)= prot; 
      omcop  (inistep,:,:)= om;
     
      % determine and save training set performances 
      % [costf,~,marg,score] = compute_costs(fvec,lbl,prot,plbl,om,mu);
      % te(inistep+1) = sum(marg>0)/nfv;
      % cf(inistep+1) = costf;

       % compute training set errors and cost function values
       % for icls=1:ncls;
       %   % compute class-wise errors (positive margin = error)
       %   cw(inistep+1,icls) = sum(marg(lbl==icls)>0)/sum(lbl==icls);
       % end;
  end;

for jstep=(ncop+1):totalsteps;    
 % calculate mean positions over latest steps
 protmean = squeeze(mean(protcop,1)); 
 ommean = squeeze(mean(omcop,1));
 ommean=ommean/sqrt(sum(sum(abs(ommean).^2))); 
 % note: normalization does not change cost function value
 %       but is done here for consistency

% compute cost functions for mean prototypes, mean matrix and both 
[costmp,~,~,score] = compute_costs(fvec,lbl,protmean,plbl,om, 0);
[costmm,~,~,score] = compute_costs(fvec,lbl,prot,    plbl,ommean,mu); 
% [costm, ~,~,score ] = compute_costs(fvec,lbl,protmean,plbl,ommean,mu); 

% remember old positions for Papari procedure
ombefore=om; 
protbefore=prot;
 
 % perform next step and compute costs etc.
[prot,om]= do_batchstep (fvec,lbl,prot,plbl,om,etap,etam,mu,mode);

% by default, step sizes are increased in every step
 etam=etam*incfac; % (small) increase of step sizes
 etap=etap*incfac; % at each learning step to enforce oscillatory behavior 

% costfunction values to compare with for Papari procedure
% evaluated w.r.t. changing only matrix or prototype
[costfp,~,~,score] = compute_costs(fvec,lbl,prot,plbl,ombefore,0);
[costfm,~,~,score] = compute_costs(fvec,lbl,protbefore,plbl,om,mu); 
   
% heuristic extension of Papari procedure
% treats matrix and prototype step sizes separately
 if (costmp <= costfp ); % decrease prototype step size and jump
                         % to mean prototypes
     etap = etap/decfac;
     prot = protmean;
 end; 
 if (costmm <= costfm ); % decrease matrix step size and jump
                         % to mean matrix
     etam = etam/decfac;
     om = ommean;   
 end
 
 % update the copies of the latest steps, shift stack of stored configs. 
 % plenty of room for improvement, I guess ...
 for iicop = 1:ncop-1;
   protcop(iicop,:,:)=protcop(iicop+1,:,:);
   omcop(iicop,:,:)  =omcop  (iicop+1,:,:);
 end;
 protcop(ncop,:,:)=prot;  omcop(ncop,:,:)=om;
 
 % determine training and test set performances 
 % here: costfunction without penalty term! 
% [costf0,~,marg,score] = compute_costs(fvec,lbl,prot,plbl,om,0);
 
 % compute total and class-wise training set errors
 % te(jstep+1) = sum(marg>0)/nfv;
 % cf(jstep+1) = costf0;
 % for icls=1:ncls;
 %     cw(jstep+1,icls) = sum(marg(lbl==icls)>0)/sum(lbl==icls);
 % end;
 
end;   % totalsteps training steps performed

%if the data was z transformed then also save the inverse prototypes,
%actually it is not necessary since the mf and st are returned.
%if doztr == 1
%    protsInv = do_inversezscore(prot, mf, st);
%else
     protsInv = prot;
%end

lambda=om'*om;   % actual relevance matrix
% define structures corresponding to the trained system and training curves
gmlvq_system =    struct('protos',prot, 'protosInv',protsInv,'lambda',lambda,'plbl',plbl,...
                         'mean_features',mf,'std_features',st);
% training_curves = struct('costs',cf,'train_error',te,...
%                          'class_wise',cw,'auroc',auc);
% param_set = struct('totalsteps',totalsteps,'doztr',...
%           doztr,'mode',mode,'rndinit',rndinit,...
%           'etam0',etam0,'etap0',etap0,'etamfin',etam,'etapfin',etap,...
%           'mu',mu,'decfac',decfac,'infac',incfac,'ncop',ncop,...
%           'rngseed',rngseed);