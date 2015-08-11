function output=sigmacr(zlens,zs)

%critical surface density 

G=4.302e-3; %pc (Mdot)^-1 (kms^-1)^2 
light=299792.458; %kms^-1

zmax=10.; %formally infinity 
zmin=zlens;
[dc, dlens]=distance(0.,zlens);
dlens=dlens.*10.^6;

%full case with pz
%output=integral(@(z)sigint(zs,zlens,z),zmin,zmax,'AbsTol',1e-6);

%delta case
[dc, dlensource]=distance(zlens,zs);
[dc, dsource]=distance(0.,zs);     
output=(dlensource)./(dsource); 

output=output.*(4.*pi.*G.*dlens/(light.*light));
output=1./output; %formula is for inverse


end

function iout=sigint(zs,zlens,z) 

    [dc, dlensource]=distance(zlens,z);
    [dc, dsource]=distance(0.,z);    
    iout=(dlensource)./(dsource).*pzsource(zs,z);
    
end

function pzs=pzsource(zs,z) 
%delta for now 

if (size(z,1)==size(z,2) && size(z,1)==1) % scalar
    if (z>zs && z<zs+0.1), pzs=1.; end
else
    pzs=zeros(1,size(z,2));
    pzs(find(z>zs & z<zs+0.1))=1.;
end

end
