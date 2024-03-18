function RMS = RMS(realu,estimu)
    [nz nx]=size(realu);
    RMS=sum(sum((estimu-realu).^2.));
    RMS =sqrt(RMS/(nz*nx));

