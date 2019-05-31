%% USING THE MATRIX FUNCTION ROUTINE TO DENOISE THE IMAGE BEFORE:

clear; clc; close all;

warning('off','all');
clear classes
cd('AIRToolsII/');
AIRToolsII_setup;
cd('..');
cd('IRtools/');
IRtools_setup;
cd('..');
warning('on','all');

FID = fopen('results_IRfun.dat','a');
fprintf(FID,'Experiment run on %s\n',datestr(now));
levelblur = {'mild','medium','severe'};

for imagename = {'satellite','pattern1','pattern2'}
    for blurindex = [1,2,3]
        fprintf(FID,'TEST CASE %s - BlurLevel %s \n',imagename{1},levelblur{blurindex});
        %% Generate the blurred and noised signal:
        options = struct('trueImage', imagename{1},...
            'BlurLevel', levelblur{blurindex}, ... % must be 'mild' or 'medium' or 'severe'.
            'BC', 'zero',...
            'CommitCrime', 'off');
        [A, bl, x_true, ProbInfo] = PRblurgauss(options);
        
        PRshowb(bl,ProbInfo,blurindex+10);
        
        for NoiseLevel = [1e-4,1e-3,1e-2,1e-1,2e-1,5e-1,6e-1]
            fprintf(FID,'%1.1e',NoiseLevel);
            [b, NoiseInfo] = PRnoise(bl, 'gauss',NoiseLevel);
            
            figure(1)
            subplot(3,3,1);
            imshow(reshape(x_true,sqrt([size(x_true,1),size(x_true,1)])),[]);
            title('True Image');
            subplot(3,3,2);
            imshow(reshape(bl,sqrt([size(x_true,1),size(x_true,1)])),[]);
            title('Blurred Image');
            subplot(3,3,3);
            imshow(reshape(b,sqrt([size(x_true,1),size(x_true,1)])),[]);
            title('Noisy+Blurred Image');
            xlabel(sprintf('PSNR %1.2f',psnr(b,x_true)));
            
            
            %% Apply the CGLS to restore the signal:
            fprintf('CGLS Algorithm:\n');
            options  = IRcgls('defaults');
            options.x_true = x_true;
            options.NoiseLevel = NoiseLevel;
            options.verbosity = 'off';
            options.IterBar = 'off';
            tic;
            [Xcgls,infocgls] = IRcgls(A,b,options);
            timecgls = toc;
            
            options.NoStop = 'on';
            
            [~,infocglsbest] = IRcgls(A,b,options);
            
            fprintf(FID, ' & %d (%d) & %1.1e & %1.2f (%1.2f)',...
                infocgls.StopReg.It,infocglsbest.BestReg.It,timecgls,...
                psnr(reshape(infocgls.StopReg.X,size(x_true)),x_true),...
                psnr(reshape(infocglsbest.BestReg.X,size(x_true)),x_true));
            
            subplot(3,3,4);
            imshow(reshape(infocgls.StopReg.X,sqrt([size(x_true,1),size(x_true,1)])),[]);
            title('CGLS');
            xlabel(sprintf('PSNR %1.2f',psnr(infocgls.StopReg.X,x_true)));
            
            %% Apply the Hybrid GMRES Algorithm
            fprintf('\nHybrid GMRES Algorithm:\n')
            
            options  = IRhybrid_gmres('defaults');
            options.x_true = x_true;
            options.NoiseLevel = NoiseLevel;
            options.verbosity = 'off';
            options.IterBar = 'off';
            tic;
            [Xgmres,infogmres] = IRhybrid_gmres(A,b,options);
            timegmres = toc;
            
            options.NoStop = 'on';
            [~,infogmresbest] = IRhybrid_gmres(A,b,options);
            
            fprintf(FID,' & %d (%d) & %1.1e & %1.2f (%1.2f)',...
                infogmres.StopReg.It,infogmresbest.BestReg.It,...
                timegmres,...
                psnr(reshape(infogmres.StopReg.X,size(x_true)),x_true),...
                psnr(reshape(infogmresbest.BestReg.X,size(x_true)),x_true));
            
            subplot(3,3,7);
            imshow(reshape(infogmres.StopReg.X,sqrt([size(x_true,1),size(x_true,1)])),[]);
            title('Hybrid GMRES');
            xlabel(sprintf('PSNR %1.2f',psnr(infogmres.StopReg.X,x_true)));
            
            %% Apply the Range-Restricted GMRES
            fprintf('\nRangeRestricted GMRES Algorithm:\n');
            
            options  = IRrrgmres('defaults');
            options.x_true = x_true;
            options.NoiseLevel = NoiseLevel;
            options.verbosity = 'off';
            options.IterBar = 'off';
            tic;
            [Xrrgmres,inforrgmres] = IRrrgmres(A,b,options);
            timerrgmres = toc;
            
            options.NoStop = 'on';
            [~,inforrgmresbest] = IRrrgmres(A,b,options);
            
            fprintf(FID,' & %d (%d) & %1.1e & %1.2f (%1.2f)', ...
                inforrgmres.StopReg.It,...
                inforrgmresbest.BestReg.It,timerrgmres,...
                psnr(reshape(inforrgmres.StopReg.X,size(x_true)),x_true),...
                psnr(reshape(inforrgmresbest.BestReg.X,size(x_true)),x_true));
            
            
            subplot(3,3,8);
            imshow(reshape(inforrgmres.StopReg.X,sqrt([size(x_true,1),size(x_true,1)])),[]);
            title('RR-GMRES');
            xlabel(sprintf('PSNR %1.2f',psnr(inforrgmres.StopReg.X,x_true)));
            
            %% CGLS for Tikhonov method
            fprintf('\nTikhonov CGLS method:\n');
            options  = IRcgls('defaults');
            figure(2)
            options.x_true = x_true;
            options.verbosity = 'off';
            options.IterBar = 'off';
            tic;
            options.RegParam = settikhonov(A,b);
            [Xtikhonov,infotikhonov] = IRcgls(A,b,options);
            timetikhonov = toc;
            
            fprintf(FID,'& %1.2e & %d & %1.1e & %1.2f',...
                options.RegParam,infotikhonov.StopReg.It,timetikhonov,...
                psnr(reshape(infotikhonov.StopReg.X,size(x_true)),x_true));
            
            figure(1)
            subplot(3,3,9);
            imshow(reshape(infotikhonov.StopReg.X,sqrt([size(x_true,1),size(x_true,1)])),[]);
            title('Tikhonov CGLS');
            xlabel(sprintf('PSNR %1.2f',psnr(infotikhonov.StopReg.X,x_true)));
            
            %% Apply the matrix-function denoiser:
            fprintf('\nMatrix Function Regularizer:\n');
            
            options  = IRfun('defaults');
            options.RegParam = NoiseLevel*1e-1;
            options.x_true = x_true;
            options.NoStop = 'off';
            options.RegBeta = 1e+6;
            options.verbosity = 0;
            options.NoiseLevel = NoiseLevel;
            options.MaxIter = 100;
            options.IterBar = 'off';
            options.Reorth = 'on';
            options.eta = 1.01;
            fprintf(FID,' & %1.1e',options.RegParam);
            
            tic;
            [X,infofun] = IRfun(A,b,options);
            timefun = toc;
            
            options.NoStop = 'on';
            [~,infofunbest] = IRfun(A,b,options);
            
            fprintf(FID,'& %d (%d) & %1.1e & %1.2f (%1.2f)',...
                infofun.StopReg.It,infofunbest.BestReg.It,timefun,...
                psnr(reshape(infofun.StopReg.X,size(x_true)),x_true),...
                psnr(reshape(infofunbest.BestReg.X,size(x_true)),x_true));
            
            subplot(3,3,5);
            imshow(reshape(infofun.StopReg.X,sqrt([size(x_true,1),size(x_true,1)])),[]);
            title('Matrix Function');
            xlabel(sprintf('PSNR %1.2f',psnr(infofun.StopReg.X,x_true)));
            
            fprintf(FID,'\\\\\n');
        end
        fprintf(FID,'\n\n\n\n');
    end
end
fclose(FID);


