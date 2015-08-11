function Hz=hubble(z)
%1/Hubble parameter
    
h0=0.7; 
om=0.3;
od=0.7;
w0=-1.0;
wa=0.0;

if (w0~=-1. || wa~=0.) 
    %w integral
    wint=0.;
    nz=1000;
    dz=(z-0.0)/(nz-1.);
    for i=1:nz
        zdash=0.+(i-1.)*dz;    
        wint=wint+(dz/(1.+zdash)).*(1.+w0+wa.*z/(1.+z));
    end
else 
   wint=0.;
end

Hz=100.*h0*sqrt(om*(1.+z).^(3.)+od*exp(3.*wint)+(1.-om-od).*(1.+z).^(2.));

Hz=1./Hz;

end


