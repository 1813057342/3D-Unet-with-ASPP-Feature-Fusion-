
clc;
clear
close all
%%
num=6400;
YTest=zeros(256,128,88,1,num);
    for i=1:num % 这里是频率
        % filename=['E:\合成资料与实际资料的三维网络与对比试验程序\正演的数据以及真实的纯信号和所对应的含噪音数据\实际微地震信号\三维时频域数据\标签数据\',num2str(i,'%.4d'),'.mat'];
        filename=['E:\合成资料与实际资料的三维网络与对比试验程序\正演的数据以及真实的纯信号和所对应的含噪音数据\训练样本\三维时频域数据\三维时频数据体输入\三维标签\',num2str(i,'%.4d'),'.mat'];
                load(filename);
        YTest(:,:,:,1,i-619)=mod(:,:,:);
        disp(i);
    end
XTest=zeros(256,128,88,1,num);
    for i=1:num% 这里是频率
        % filename=['E:\合成资料与实际资料的三维网络与对比试验程序\正演的数据以及真实的纯信号和所对应的含噪音数据\实际微地震信号\三维时频域数据\训练样本\',num2str(i,'%.4d'),'.mat'];
       filename=['E:\合成资料与实际资料的三维网络与对比试验程序\正演的数据以及真实的纯信号和所对应的含噪音数据\训练样本\三维时频域数据\三维时频数据体输入\三维训练\',num2str(i,'%.4d'),'.mat'];
                load(filename);
        XTest(:,:,:,1,i-619)=mod(:,:,:);
        disp(i);
    end
    %%
    % 验证数据的准确性
    train1=XTest(:,:,1:44,1,1);
    train2=XTest(:,:,45:88,1,1);
    train3=complex(train1,train2);
    mod=abs(train3);
    x = 1:size(mod,2);%频率
    y = 1:size(mod,1);%时间
    z = 1:size(mod,3);%道数
    [xx,yy,zz] = meshgrid(x,y,z);
    xslice = [];
    yslice = [];
    zslice = [1:1:45];
    figure;
    slice(xx,yy,zz,mod,xslice,yslice,zslice);
    colorbar;colormap(jet);
    axis equal;
    caxis([0,1]);
    shading interp;
    xlabel('频率');
    ylabel('时间');
    zlabel('道数');
    alpha color
    alpha scaled
    save('mic_3D_time_frequency140.mat',...                                             %保存部分数据
    'XTest','YTest');
%%
AllXTrain=train;                                                          %AllXTrain训练集
AllYTrain=label;                                                          %AllYTrain标签集
clear label train
AllNumTrain=size(AllXTrain,5);
FirTraPer=0.85;                                                             %训练集的比例
FirValPer=0.1;                                                             %验证集的比例
FirTesPer=0.05;                                                             %测试集的比例
FirTesNum=floor(AllNumTrain*FirTesPer);                                    %训练集的数量
FirValNum=floor(AllNumTrain*FirValPer);                                    %验证集的数量
FirTraNum=AllNumTrain-FirTesNum-FirValNum;                                 %测试集的数量

XTest      = zeros(size(AllXTrain,1),size(AllXTrain,2),size(AllXTrain,3),1,FirTesNum);
XValidation= zeros(size(AllXTrain,1),size(AllXTrain,2),size(AllXTrain,3),1,FirValNum);
XTrain     = zeros(size(AllXTrain,1),size(AllXTrain,2),size(AllXTrain,3),1,FirTraNum);
YTest      = zeros(size(AllXTrain,1),size(AllXTrain,2),size(AllXTrain,3),1,FirTesNum);
YValidation= zeros(size(AllXTrain,1),size(AllXTrain,2),size(AllXTrain,3),1,FirValNum);
YTrain     = zeros(size(AllXTrain,1),size(AllXTrain,2),size(AllXTrain,3),1,FirTraNum);

XTest(:,:,:,1,1:FirTesNum)       = AllXTrain(:,:,:,1,1:FirTesNum );
XValidation(:,:,:,1,1:FirValNum) = AllXTrain(:,:,:,1,FirTesNum+1:FirValNum+FirTesNum);
XTrain(:,:,:,1,1:FirTraNum)      = AllXTrain(:,:,:,1,FirTesNum+FirValNum+1:AllNumTrain);
clear AllXTrain
YTest(:,:,:,1,1:FirTesNum)       = AllYTrain(:,:,:,1,1:FirTesNum );
YValidation(:,:,:,1:FirValNum) = AllYTrain(:,:,:,1,FirTesNum+1:FirValNum+FirTesNum);
YTrain(:,:,:,1,1:FirTraNum)      = AllYTrain(:,:,:,1,FirTesNum+FirValNum+1:AllNumTrain);
clear AllYTrain
save('microseismic_3D_time_frequency.mat',...                                             %保存部分数据
    'XTrain','YTrain','XValidation','YValidation', 'XTest','YTest');
return








