function snr = snr(seismo_clean,seismo_noise)
    snr = 0;
    [nt nx]=size(seismo_clean);
    seismo_clean_power = sum(sum(seismo_clean.^2))/(nt*nx);
    seismo_noise_power = sum(sum((seismo_noise-seismo_clean).^2))/(nt*nx);
    snr = 10*log10(seismo_clean_power/seismo_noise_power);
