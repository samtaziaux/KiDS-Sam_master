
% tom kitching 
% code to compute \Delta\Sigma 

%%
filename='data_all.dat';  %list of RA and Dec

%%
%load the GAMA 
g=dlmread('GAMA_CP_4Tom_NH.dat');

%%
nlens=size(g,1);
fprintf(' nlens=%d\n',nlens);

%load the kids catalogue 
%SeqNr ALPHA_J2000 DELTA_J2000 e1_A e2_A MAG_GAAP_u MAG_GAAP_g MAG_GAAP_r MAG_GAAP_i PSF_e1 PSF_e2 weight Z_B m_cor c2_A
load('~/Desktop/KiDS_dr2/source.mat');
m=p;
nobj=size(m,1);
e1=m(:,4);
e2=m(:,5);
zsource=m(:,13);
weightsource=m(:,12);
msource=m(:,15);

%% loop over lenses and stack the mean tangential and cross signals
nbin=20;
y=zeros(nlens,nbin);
z=zeros(nlens,nbin);
w=zeros(nlens,nbin);
k=zeros(nlens,nbin);

thetamin=1e-2;%comoving Mpc 
thetamax=10.;

%for covariance 
xcsi=zeros(nobj,nbin);
xssi=zeros(nobj,nbin);
xzsi=zeros(nobj,nbin);

x=10.^(log10(thetamin)+((1:nbin)-1).*(log10(thetamax)-log10(thetamin))./(nbin-1.));

load('dist.mat');
global distarray redsarray;

for i=1:nlens
    
fprintf(' %d\n',i); 

%find angular diameter distance 
[dc,da]=distance(0.,g(i,4));
    
%find background galaxies around lens 
deg=(sqrt((m(:,2)-g(i,2)).^2+(m(:,3)-g(i,3)).^2)).*(pi./180.).*da; %comoving seperation 
raok=(m((deg<0.1 & zsource>g(i,4)) & zsource<3.0,2)-g(i,2)).*(pi./180.).*da; %all within a max angle degree, and zeroed
deok=(m((deg<0.1 & zsource>g(i,4)) & zsource<3.0,3)-g(i,3)).*(pi./180.).*da;

if (isempty(raok)==0 && isempty(deok)==0)%check that there are galaxies around the lens

ok=(msource>-80 & deg<thetamax & zsource>g(i,4) & zsource<3.0);   
    
raok=(m(ok,2)-g(i,2)).*(pi./180.).*da; %all within a max angle degree, and zeroed
deok=(m(ok,3)-g(i,3)).*(pi./180.).*da;
e1ok=e1(ok);
e2ok=e2(ok);
weightsourceok=weightsource(ok);
zsourceok=zsource(ok);
msourceok=msource(ok);
indexok=m(ok,1);

sigmarok=zeros(size(zsourceok,1),1);

for j=1:size(zsourceok,1)
    sigmarok(j)=sigmacr(g(i,4),zsourceok(j)); %outputs sigma NOT sigma^-1
end

weightsourceok=weightsourceok./sigmarok./sigmarok; %\tilde wls

%** loop over projected distance bins
theta=sqrt(raok.^2+deok.^2);
for j=1:nbin

    thetabin=10.^(log10(thetamin)+(j-1).*(log10(thetamax)-log10(thetamin))./(nbin-1.)); %logarithmically spaced
    thetabinupper=10.^(log10(thetamin)+(j).*(log10(thetamax)-log10(thetamin))./(nbin-1.)); %logarithmically spaced
    
    inbin=find(theta>=thetabin & theta<thetabinupper);
    
    if (size(inbin,1)~=0) 
    e1_inbin=e1ok(inbin);
    e2_inbin=e2ok(inbin);
    pa_inbin=atan(deok(inbin)./raok(inbin)); %projected_angles (ratio so angle to comoving conversion drops out)
    
    et_inbin=-(e1_inbin.*cos(2.*pa_inbin)+e2_inbin.*sin(2.*pa_inbin));
    ex_inbin= (e2_inbin.*sin(2.*pa_inbin)-e2_inbin.*cos(2.*pa_inbin));
   
    y(i,j)=sum(et_inbin.*weightsourceok(inbin).*sigmarok(inbin));
    z(i,j)=sum(ex_inbin.*weightsourceok(inbin).*sigmarok(inbin));
    w(i,j)=sum(weightsourceok(inbin)); %wls\sigma^-2
    k(i,j)=sum(weightsourceok(inbin).*msourceok(inbin));
      
    csiadd=(1./sigmarok(inbin)).*cos(2.*pa_inbin);
    ssiadd=(1./sigmarok(inbin)).*sin(2.*pa_inbin);
    zsiadd=(1./sigmarok(inbin)./sigmarok(inbin));
    
    if (isempty(find(isnan(csiadd))==1)==1), xcsi(indexok(inbin),j)=xcsi(indexok(inbin),j)-csiadd; end
    if (isempty(find(isnan(csiadd))==1)==1), xssi(indexok(inbin),j)=xssi(indexok(inbin),j)-ssiadd; end
    if (isempty(find(isnan(csiadd))==1)==1), xzsi(indexok(inbin),j)=xzsi(indexok(inbin),j)+zsiadd; end

    end
    
end

end %isempty

end %lens

%covariance
cov=zeros(nbin,nbin);
sige=((std(e1)+std(e2))./2.);
for i=1:nbin
    for j=1:nbin
        summer=weightsource.*weightsource.*(xcsi(:,i).*xcsi(:,j)+xssi(:,i).*xssi(:,j));
        dimmei=weightsource.*xzsi(:,i);
        dimmej=weightsource.*xzsi(:,j);
        cov(i,j)=sige.*sige.*sum(summer)./(sum(dimmei).*sum(dimmej));
    end
end

%print r sigma and diagonal of covariance
f=fopen(filename,'w');
xerr=sqrt(diag(cov));
yerr=x.*sqrt(diag(cov))';
yerr(yerr==0.)=999.;
xerr(xerr==0.)=999.;
yerr(isnan(yerr)==1)=999.;
xerr(isnan(xerr)==1)=999.;
for i=1:nbin
fprintf(f,'%f %f %f %f %f\n',x(i),meancrossx(i),x(i).*meansignal(i),xerr(i),yerr(i));
end
fclose(f);

