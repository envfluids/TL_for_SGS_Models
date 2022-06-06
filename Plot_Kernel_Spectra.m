clc,clear

re_str = 'Re_1k_to_Re_100k';
kf_str = 'nxy_25_to_nxy_4';
decay = 'decay_to_forced_64';

for l = 1:11
    load(['Weights_TL_from_' kf_str '_per_train_10_layers_' num2str(l) '.mat'])
    eval(['l' num2str(l) '_w_TL = l' num2str(l) '_w;'])
    if l == 1
        l11 = l11_w;
    end
end
l11_w = l11;
clear('l11')

%%

NX = 128;

Lx = 2*pi;
% Wavenumbers
dx = Lx/NX;
kx = (2*pi/Lx)*[(-NX/2+1):(NX/2)];
[Kx,Ky] = meshgrid(kx,kx);
%%

kernels = l10_w;
kernels_TL = l10_w_TL;

changes = zeros(1,64^2);
changes_spec = zeros(1,64^2);

for i = 1:64
    for j = 1:64
        changes(i+(j-1)*64) = norm(squeeze(kernels(:,:,i,j)-...
            kernels_TL(:,:,i,j)),'fro');
        
        ftt_kernel = fft2(squeeze(kernels(:,:,i,j)),128,128);
        ftt_kernel_TL = fft2(squeeze(kernels_TL(:,:,i,j)),128,128);
        
        changes_spec(i+(j-1)*64) = norm(abs(ftt_kernel)...
            -abs(ftt_kernel_TL),'fro');
    end
end
chngs_lim = prctile(changes,99.9);
locs = find(changes>chngs_lim);
chngs_spec_lim = prctile(changes_spec,99.9);
locs_spec = find(changes_spec>chngs_spec_lim);

%%

for k = 3
    ind_i = mod(locs_spec(k),64);
    if ind_i == 0
        ind_i = 64;
    end
    ind_j = floor(locs_spec(k)/64)+1;
    
    ftt_kernel = fft2(squeeze(kernels(:,:,ind_i,ind_j)),128,128);
    ftt_kernel_TL = fft2(squeeze(kernels_TL(:,:,ind_i,ind_j)),128,128);


    figure
    contourf(Kx,Ky,abs(fftshift(ftt_kernel)),128,'LineStyle','none')
    xticklabels({})
    yticklabels({})
    set(gca,'YColor','none','Xcolor','none','Box','off','lineWidth',10,...
        'FontName','CMU Serif')
    title('BNN')
    yticks([-60 -30 0 30 60])
    xticks([-60 -30 0 30 60])
    axis equal
    caxis([0 .5])

    m = redblue();
    colormap(m)
    
    
    figure
    contourf(Kx,Ky,abs(fftshift(ftt_kernel_TL)),128,'LineStyle','none')
    xticklabels({})
    yticklabels({})
    set(gca,'YColor','none','Xcolor','none','Box','off','lineWidth',10,...
    'FontName','CMU Serif')

    yticks([-60 -30 0 30 60])
    xticks([-60 -30 0 30 60])
    axis equal
    caxis([0 .5])
    m = redblue();
    colormap(m)   
    title('TLNN')

end

% saveas(gcf,'Change_Filter_Layer_2_Decay_TL.png')



