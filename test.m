img1 = imread('1-u.png');

img2 = imread('1-g.jpg');

img2 = imresize(img2, size(img1,1,2));

psnr_value = psnr(img1, img2);

ssim_value = ssim(img1, img2);

mse_value = immse(img1, img2);

mae_value = mean(abs(double(img1(:)) - double(img2(:))));

fprintf('PSNR: %.4f\n', psnr_value);
fprintf('SSIM: %.4f\n', ssim_value);
fprintf('MSE: %.4f\n', mse_value);
fprintf('MAE: %.4f\n', mae_value);