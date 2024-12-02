clc;
clear;
close all;
%% 创建写入数据的文件夹
tic;
now=fix(clock);%读取当前时间
folder_name=[num2str(now(1)), '_',num2str(now(2)), '_',num2str(now(3)), '_',num2str(now(4)), '_',num2str(now(5)), '_',num2str(now(6)),'550样本卷积核消融实验3dASPP-UNET'];
% folder_name=[date, '-',num2str(now(4)), '-',num2str(now(5)), '-',num2str(now(6))]; %创建文件名
mkdir(folder_name);    %mkdir函数是创建文件夹       
%创建时间文件夹
% folder_name_net=[folder_name,'_NetLayerIteration'];                      %创建迭代网络保存的文件夹名
% mkdir (folder_name_net);                                                 %创建存储迭代过程中网络的文件夹
currentFolder = pwd;   %返回当前文件夹
currentFolder2 = [currentFolder,'\',folder_name];                          %存入文件夹路径：windows'\',linux'/'
%%
 load('microseismic_3D_time_frequency.mat');
CNN_Time01=clock;
%%%%%%%First:串行层搭建Unet主要层,替换最后两层，将语义分割转换为回归网络%%%%%%%
SizeXTrain=size(XTrain);                                                   %获得输入参数规模
heightInput =SizeXTrain(1,1);
widthInput = SizeXTrain(1,2);
depthInput = SizeXTrain(1,3);

%%
lgraph = layerGraph();
tempLayers = [
    image3dInputLayer([256 128 88 1],"Name","ImageInputLayer")
    convolution3dLayer([5 5 5],16,"Name","Encoder-Stage-1-Conv-1","Padding","same")
    batchNormalizationLayer("Name","Encoder-Stage-1-BN-1")
    leakyReluLayer(0.01,"Name","Encoder-Stage-1-ReLU-1")
    convolution3dLayer([3 3 3],16,"Name","Encoder-Stage-1-Conv-2","Padding","same")
    batchNormalizationLayer("Name","Encoder-Stage-1-BN-2")
    leakyReluLayer(0.01,"Name","Encoder-Stage-1-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-1-MaxPool","Padding","same","Stride",[2 2 2])
    convolution3dLayer([5 5 5],32,"Name","Encoder-Stage-2-Conv-1","Padding","same")
    batchNormalizationLayer("Name","Encoder-Stage-2-BN-1")
    leakyReluLayer(0.01,"Name","Encoder-Stage-2-ReLU-1")
    convolution3dLayer([3 3 3],32,"Name","Encoder-Stage-2-Conv-2","Padding","same")
    batchNormalizationLayer("Name","Encoder-Stage-2-BN-2")
    leakyReluLayer(0.01,"Name","Encoder-Stage-2-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-2-MaxPool","Padding","same","Stride",[2 2 2])
    convolution3dLayer([5 5 5],64,"Name","Encoder-Stage-3-Conv-1","Padding","same")
    batchNormalizationLayer("Name","Encoder-Stage-3-BN-1")
    leakyReluLayer(0.01,"Name","Encoder-Stage-3-ReLU-1")
    convolution3dLayer([3 3 3],64,"Name","Encoder-Stage-3-Conv-2","Padding","same")
    batchNormalizationLayer("Name","Encoder-Stage-3-BN-2")
    leakyReluLayer(0.01,"Name","Encoder-Stage-3-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-3-MaxPool","Padding","same","Stride",[2 2 2])
    convolution3dLayer([3 3 3],128,"Name","Encoder-Stage-4-Conv-1","Padding","same")
    batchNormalizationLayer("Name","Encoder-Stage-4-BN-1")
    leakyReluLayer(0.01,"Name","Encoder-Stage-4-ReLU-1")
    convolution3dLayer([5 5 5],128,"Name","Encoder-Stage-4-Conv-2","Padding","same")
    batchNormalizationLayer("Name","Encoder-Stage-4-BN-2")
    leakyReluLayer(0.01,"Name","Encoder-Stage-4-ReLU-2")
    dropoutLayer(0.01,"Name","drop001")
    transposedConv3dLayer([2 2 2],64,"Name","Decoder-Stage-1-UpConv","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([3 3 3],32,"Name","res5a_branch1","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([7 7 7],32,"Name","conv1","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","bn_conv1")
    reluLayer("Name","conv1_relu")
    convolution3dLayer([3 3 3],32,"Name","res2b_branch2b","Padding",[1 1 1;1 1 1])
    batchNormalizationLayer("Name","bn2a_branch2b2")
    reluLayer("Name","res2b_relu")
    transposedConv3dLayer([3 3 3],32,"Name","transposed-conv3d","Cropping","same","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","res4b_relu_1")
    batchNormalizationLayer("Name","bn5a_branch1")
    convolution3dLayer([3 3 3],32,"Name","conv3d","Padding","same")
    reluLayer("Name","res5a_relu")
    convolution3dLayer([3 3 3],32,"Name","res5b_branch2a","DilationFactor",[2 2 2],"Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2a")
    reluLayer("Name","res5b_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],64,"Name","aspp_Conv_1","Padding","same")
    batchNormalizationLayer("Name","aspp_BatchNorm_1")
    reluLayer("Name","aspp_Relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],64,"Name","aspp_Conv_2","DilationFactor",[6 6 6],"Padding","same")
    batchNormalizationLayer("Name","aspp_BatchNorm_2")
    reluLayer("Name","aspp_Relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],64,"Name","aspp_Conv_3","DilationFactor",[12 12 12],"Padding","same")
    batchNormalizationLayer("Name","aspp_BatchNorm_3")
    reluLayer("Name","aspp_Relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],64,"Name","aspp_Conv_4","DilationFactor",[18 18 18],"Padding","same")
    batchNormalizationLayer("Name","aspp_BatchNorm_4")
    reluLayer("Name","aspp_Relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","catAspp")
    convolution3dLayer([1 1 1],32,"Name","dec_c1","WeightLearnRateFactor",10)
    convolution3dLayer([1 1 1],32,"Name","dec_c1_1","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat")
    convolution3dLayer([3 3 3],128,"Name","dec_c4","Padding","same")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","Decoder-Stage-1-DepthConcatenation")
    convolution3dLayer([3 3 3],64,"Name","Decoder-Stage-1-Conv-1","Padding","same")
    batchNormalizationLayer("Name","Decoder-Stage-1-BN-1")
    leakyReluLayer(0.01,"Name","Decoder-Stage-1-ReLU-1")
    convolution3dLayer([5 5 5],64,"Name","Decoder-Stage-1-Conv-2","Padding","same")
    batchNormalizationLayer("Name","Decoder-Stage-1-BN-2")
    leakyReluLayer(0.01,"Name","Decoder-Stage-1-ReLU-2")
    transposedConv3dLayer([2 2 2],32,"Name","Decoder-Stage-2-UpConv","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-2-DepthConcatenation")
    convolution3dLayer([3 3 3],32,"Name","Decoder-Stage-2-Conv-1","Padding","same")
    batchNormalizationLayer("Name","Decoder-Stage-2-BN-1")
    leakyReluLayer(0.01,"Name","Decoder-Stage-2-ReLU-1")
    convolution3dLayer([5 5 5],32,"Name","Decoder-Stage-2-Conv-2","Padding","same")
    batchNormalizationLayer("Name","Decoder-Stage-2-BN-2")
    leakyReluLayer(0.01,"Name","Decoder-Stage-2-ReLU-2")
    transposedConv3dLayer([2 2 2],16,"Name","Decoder-Stage-3-UpConv","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-3-DepthConcatenation")
    convolution3dLayer([3 3 3],16,"Name","Decoder-Stage-3-Conv-1","Padding","same")
    batchNormalizationLayer("Name","Decoder-Stage-3-BN-1")
    leakyReluLayer(0.01,"Name","Decoder-Stage-3-ReLU-1")
    convolution3dLayer([3 3 3],16,"Name","Decoder-Stage-3-Conv-2","Padding","same")
    convolution3dLayer([3 3 3],8,"Name","Decoder-Stage-3-Conv-3","Padding","same")
    convolution3dLayer([3 3 3],4,"Name","Decoder-Stage-3-Conv-4","Padding","same")
    convolution3dLayer([3 3 3],4,"Name","Decoder-Stage-3-Conv-5","Padding","same")
    convolution3dLayer([3 3 3],1,"Name","Decoder-Stage-3-Conv-6","Padding","same")
    regressionLayer("Name","mean-squared-error")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Decoder-Stage-3-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Decoder-Stage-2-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Encoder-Stage-3-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","res5a_branch1");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Decoder-Stage-1-DepthConcatenation/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpConv","Decoder-Stage-1-DepthConcatenation/in1");
lgraph = connectLayers(lgraph,"res5a_branch1","conv1");
lgraph = connectLayers(lgraph,"res5a_branch1","res4b_relu_1");
lgraph = connectLayers(lgraph,"transposed-conv3d","depthcat/in2");
lgraph = connectLayers(lgraph,"res5b_relu","aspp_Conv_1");
lgraph = connectLayers(lgraph,"res5b_relu","aspp_Conv_2");
lgraph = connectLayers(lgraph,"res5b_relu","aspp_Conv_3");
lgraph = connectLayers(lgraph,"res5b_relu","aspp_Conv_4");
lgraph = connectLayers(lgraph,"aspp_Relu_1","catAspp/in1");
lgraph = connectLayers(lgraph,"aspp_Relu_2","catAspp/in2");
lgraph = connectLayers(lgraph,"aspp_Relu_3","catAspp/in3");
lgraph = connectLayers(lgraph,"aspp_Relu_4","catAspp/in4");
lgraph = connectLayers(lgraph,"dec_bn1","depthcat/in1");
lgraph = connectLayers(lgraph,"relu_2","Decoder-Stage-1-DepthConcatenation/in3");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpConv","Decoder-Stage-2-DepthConcatenation/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpConv","Decoder-Stage-3-DepthConcatenation/in1");
plot(lgraph);

%%
miniBatchSize=8;%训练迭代的最小批次大小8
MaxEpoch=50;
ValTimesPerEpoch=10;                                                               %每一个epcoh需要验证的次数，最好整除
validationFrequency=floor( SizeXTrain(1,5)/ (miniBatchSize*ValTimesPerEpoch) );   %设置恰当的ValTimesPerEpoch，最好整除
ValPatNepoch=5;                                                                  %N个epoch停止训练
ValPat=ValPatNepoch*ValTimesPerEpoch;
initiallearnrate=0.0015;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize , ...
    'GradientThreshold',1, ...
    'MaxEpochs',MaxEpoch, ...
    'InitialLearnRate',initiallearnrate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',ValPat, ...
    'Plots','training-progress', ...%%绘制训练图像
    'ExecutionEnvironment','gpu',...
    'Verbose',true);%%显示训练进度


 YValidationPredicted = predict(net,XValidation,"ExecutionEnvironment","gpu"); 

%%
% ifig=1030;
% ifig=ifig+1;
% figure(ifig);
% %loss
% xtext=5;
% ytext=6;
% xylabelsize=10;
% legendfontsize=10;
% setgcafontsize=12;
% MaxIteration=size(info.TrainingLoss,2);                          %第一个CNN的总迭代次数，直接取infoFir.TrainingLoss的长度其为横坐标
% CNNepochIteration=(1:1:MaxIteration)./(size(YTrain,1)/miniBatchSize);
% ValSit=0:validationFrequency:MaxIteration;ValSit(1)=1;          %用验证频率
% figure(130)
% numoutput=SizeYTest(1,2)*SizeYTest(1,2);                               % the number piex of output
% 
% plot(ValSit ./validationFrequency,(2/numoutput)*info.TrainingLoss(ValSit(:)),...                        %all loss of rnn train set with epoch(X)
%     '-or' ,'LineWidth',1,...
%     'MarkerEdgeColor','r',...
%     'MarkerFaceColor','r',...
%     'MarkerSize',5);
% hold on
% plot(ValSit ./validationFrequency,(2/numoutput)*info.ValidationLoss(ValSit(:)),...
%     '-.ok' ,'LineWidth',1,...
%     'MarkerEdgeColor','k',...
%     'MarkerFaceColor','w',...
%     'MarkerSize',5);
% 
% xlabel('Epoch','fontsize',xylabelsize,'fontweight','b');
% ylabel('MSE loss','fontsize',xylabelsize,'fontweight','b');
% legend('train','validation','fontsize',xylabelsize)
% xlim([0 100]);
% ylim([0.01 10]);
% grid on;
% box on;
% % str = orderfig(1);
% % text(xtext,ytext,str,"FontSize",14)
% set(gca,'YScale','log','FontSize',setgcafontsize,'Fontname', 'Times New Roman');
% 
% set(130,"Units","centimeters","Position",[0 2 15 6])

%%
cd(currentFolder2)
save('deeplab_model.mat','net');
save('deeplan__model_info.mat','info');
cd(currentFolder)
%%%%%%%%%%%%%%%End Train and Prediect the CNN-Unet Modification%%%%%%%%%%%%
%% 画图验证 
for i=1:4
mod=XTest(:,:,:,i);
x = 1:size(mod,2);%频率
y = 1:size(mod,1);%时间
z = 1:size(mod,3);%道数
[xx,yy,zz] = meshgrid(x,y,z);
xslice = [30*i:30*i+120];
yslice = [];
zslice = [];
figure;
title('i');
slice(xx,yy,zz,mod,xslice,yslice,zslice);
colorbar;colormap(jet);
axis equal;
caxis([-0.2,0.2]);
shading interp;
xlabel('频率');
ylabel('时间');
zlabel('道数');
alpha color ;
alpha scaled ;
end
%% 画图验证 
% for i=1:4
% mod=XValidation(:,:,:,i);
% x = 1:size(mod,2);%频率
% y = 1:size(mod,1);%时间
% z = 1:size(mod,3);%道数
% [xx,yy,zz] = meshgrid(x,y,z);
% xslice = [30*i:30*i+120];
% yslice = [];
% zslice = [];
% figure;
% title('i');
% slice(xx,yy,zz,mod,xslice,yslice,zslice);
% colorbar;colormap(jet);
% axis equal;
% caxis([-0.2,0.2]);
% shading interp;
% xlabel('频率');
% ylabel('时间');
% zlabel('道数');
% end
%% 画图验证 含噪音的训练数据
