%% TEST OF OUR CODE BUILT AS IN IRtool EXTENSION

close all;
close all hidden;
clear;
clc;

warning('off','all');
clear classes
cd('AIRToolsII/');
AIRToolsII_setup;
cd('..');
cd('IRtools/');
IRtools_setup;
cd('..');
warning('on','all');

FID = fopen('results_tomo_residual_stopping_kaczparam.dat','w');
FID2 = fopen('results_tomo_residual_stopping_parameters.dat','w');

%% ITERATION
for theproblem = 1:2
    switch theproblem
        case 1
            N = 50;
            theta = 0:1:179;
            p = round(sqrt(2)*N);
            d = p-1;
            [A,bl,x_true,theta,p,d] = paralleltomo(N,theta,p,d,0,0);
            problemname = 'paralleltomo';
        case 2
            N = 70;
            [A,bl,x_true,s,p] = seismictomo(N,N,2*N,0,0);
            problemname = 'seismictomo';
    end
    fprintf(FID,'Tomography Experiment %s launched on: %s\n',problemname,datestr(now));
    for NoiseLevel = [1e-4,1e-3,1e-2,1e-1,2e-1,5e-1,6e-1]
        close all;
        close all hidden;
        fprintf(FID,'%1.1e',NoiseLevel);
        fprintf(FID2,'%1.1e',NoiseLevel);
        rng(0);                         % Initialize random number generator.
        e = randn(size(bl));            % Gaussian white noise.
        e = NoiseLevel*norm(bl)*e/norm(e);   % Scale the noise vector.
        b = bl + e;                     % Add the noise to the pure data.
        
        
        %% MATRIX FUNCTION REGULARIZATION
        options  = IRfun('defaults');
        options.RegParam = NoiseLevel*1e-1;
        options.x_true = x_true;
        options.NoStop = 'off';
        options.RegBeta = 1e+9;
        options.eta = 1.01;
        options.NoiseLevel = NoiseLevel;
        options.MaxIter = 200;
        options.verbosity = 1;
        options.IterBar = 'on';
        options.RegType = 'normal';
            
        tic;
        [X,infofun] = IRfun(A,b,options);
        timefun = toc;
        
        options.NoStop = 'on';
        options.verbosity = 0;
        options.IterBar = 'off';
        [~,infofunbest] = IRfun(A,b,options);
        
        fprintf(FID,'& %d (%d) & %1.1e & %1.2f (%1.2f)',...
            infofun.StopReg.It,infofunbest.BestReg.It,timefun,...
            psnr(reshape(infofun.StopReg.X,size(x_true)),x_true),...
            psnr(reshape(infofunbest.BestReg.X,size(x_true)),x_true));
        
        
        
        clear options
        
        %% MAXIMUM NUMBER OF ITERATIONS FOR COMPARISON METHODS
        kmax = 800;
        
        %% KACZMARZ METHOD
        options.waitbar = false;
        options.verbose = 0;
        options.relaxpar = train_relaxpar(A,b,x_true,@kaczmarz,20,options);
        
        [Xkaczbest,infokaczbest] = ...
            kaczmarz(A,b,1:kmax,zeros(size(A(b,'transp'))),options);
        
        Ekaczrandbest = zeros(kmax,1);
        for k=1:kmax
            Ekaczrandbest(k) = psnr(reshape(Xkaczbest(:,k),size(x_true)),x_true);
        end
        [minEkacz,Kkacz] = max(Ekaczrandbest);
        
        options.stoprule.taudelta = ...
            NoiseLevel*train_dpme(A,bl,x_true,@kaczmarz,'DP',NoiseLevel,10,20,options);
        options.stoprule.type = 'DP';
        tic;
        [Xkacz,infokacz] = ...
            kaczmarz(A,b,kmax,zeros(size(A(b,'transp'))),options);
        timekacz = toc;
        
        fprintf(FID,'& %d (%d) & %1.1e & %1.2f (%1.2f)',...
            infokacz.itersaved,Kkacz,timekacz,...
            psnr(reshape(Xkacz,size(x_true)),x_true),...
            minEkacz);
        fprintf(FID2,'& %1.1e & %1.1e',...
            options.relaxpar,options.stoprule.taudelta);
        
        clear options
        
        %% RANDOM KACZMARZ METHOD
        options.waitbar = false;
        options.verbose = 0;
        options.relaxpar = train_relaxpar(A,b,x_true,@randkaczmarz,20,options);
        
        [Xkaczrandbest,infokaczbestrand] = ...
            randkaczmarz(A,b,1:kmax,zeros(size(A(b,'transp'))),options);
        
        Ekaczrandbest = zeros(kmax,1);
        for k=1:kmax
            Ekaczrandbest(k) = psnr(reshape(Xkaczrandbest(:,k),size(x_true)),x_true);
        end
        [minEkaczrandbest,Kkaczrandbest] = max(Ekaczrandbest);
        
        
        options.stoprule.taudelta = ...
            NoiseLevel*train_dpme(A,bl,x_true,@randkaczmarz,'DP',NoiseLevel,10,20,options);
        options.stoprule.type = 'DP';
        tic;
        [Xkaczrand,infokaczrand] = ...
            randkaczmarz(A,b,kmax,zeros(size(A(b,'transp'))),options);
        timekaczrand = toc;
        
        fprintf(FID,'& %d (%d) & %1.1e & %1.2f (%1.2f)',...
            infokaczrand.finaliter,Kkaczrandbest,timekaczrand,...
            psnr(reshape(Xkaczrand,size(x_true))...
            ,x_true),minEkaczrandbest);
        fprintf(FID2,'& %1.1e & %1.1e',...
            options.relaxpar,options.stoprule.taudelta);
        
        
        clear options
        
        %% CIMMINO METHOD
        relaxpar_cimmino = train_relaxpar(A,b,x_true,@cimmino,20);
        tau_cimmino = 1.01;
        
        options.relaxpar = relaxpar_cimmino;
        Xcimmino = cimmino(A,b,1:kmax);
        Ecimmino = zeros(kmax,1);
        for k=1:kmax
            Ecimmino(k) = psnr(reshape(Xcimmino(:,k),size(x_true)),x_true);
        end
        [minEcimmino,Kcimmino] = max(Ecimmino);
        
        options.stoprule.type = 'DP';
        options.stoprule.taudelta = tau_cimmino*NoiseLevel;
        [Xcimmino,infocimmino] = cimmino(A,b,1:kmax,[],options);
        
        fprintf(FID,'& %d (%d) & %1.1e & %1.2f (%1.2f)',...
            infocimmino.finaliter,Kcimmino,infocimmino.timetaken,...
            psnr(reshape(Xcimmino(:,infocimmino.finaliter),size(x_true)),x_true),...
            minEcimmino);
        fprintf(FID2,'& %1.1e & %1.1e',...
            options.relaxpar,options.stoprule.taudelta);
        
        clear options
        %% LANDWEBER METHOD
        relaxpar_landweber = train_relaxpar(A,b,x_true,@landweber,20);
        tau_landweber = 1.01;
        
        options.relaxpar = relaxpar_landweber;
        Xlandweber = landweber(A,b,1:kmax);
        Elandweber = zeros(kmax,1);
        for k=1:kmax
            Elandweber(k) = psnr(reshape(Xlandweber(:,k),size(x_true)),x_true);
        end
        [minElandweber,Klandweber] = max(Elandweber);
        
        options.stoprule.type = 'DP';
        options.stoprule.taudelta = tau_landweber*NoiseLevel;
        [Xlandweber,infolandweber] = landweber(A,b,1:kmax,[],options);
        
        fprintf(FID,'& %d (%d) & %1.1e & %1.2f (%1.2f)',...
            infolandweber.finaliter,Klandweber,infolandweber.timetaken,...
            psnr(reshape(Xlandweber(:,infolandweber.finaliter),size(x_true)),x_true),...
            minElandweber);
        fprintf(FID2,'& %1.1e & %1.1e \\\\\n',...
            options.relaxpar,options.stoprule.taudelta);
        
        clear options
        
        %% CGLS
        options  = IRcgls('defaults');
        options.MaxIter = kmax;
        options.x_true = x_true;
        options.verbosity = 'off';
        options.IterBar = 'off';
        
        [Xcglsbest,infocglsbest] = IRcgls(A,b,options);
        
        options.NoiseLevel = NoiseLevel;
        options.verbosity = 'off';
        options.IterBar = 'off';
        
        tic;
        [Xcgls,infocgls] = IRcgls(A,b,options);
        timecgls = toc;
        
        fprintf(FID, ' & %d (%d) & %1.1e & %1.2f (%1.2f)',....
            infocgls.StopReg.It,infocglsbest.BestReg.It,timecgls,...
            psnr(reshape(infocgls.StopReg.X,size(x_true)),x_true),...
            psnr(reshape(infocglsbest.BestReg.X,size(x_true)),x_true));
        
        
        fprintf(FID,'\\\\\n');
    end
end
fclose(FID);