function [dcomov,dangle]=distance(z1,z2) 
%comovoing distance in Mpc 
light=299792.458; %kms^-1

global distarray redsarray;

if (size(z2,1)==size(z2,2) && size(z2,1)==1) % scalar

    if isempty(distarray)==0 && isempty(redsarray)==0
    z1i=(int32(z1.*1000.))+1;
    z2i=(int32(z2.*1000.))+1;
    dist=distarray(z1i,z2i);
    else 
    dist=integral(@hubble,z1,z2,'AbsTol',1e-6);
    dist=light.*dist;
    end
    
    dcomov=dist;
    dangle=dist./(1.+z2);

else %array
    
  dcomov=zeros(1,size(z2,2));
  dangle=zeros(1,size(z2,2));
  for i=1:size(z2,2)
      if isempty(distarray)==0 
      z1i=(int32(z1.*1000.))+1;
      z2i=(int32(z2.*1000.))+1;
      dist=distarray(z1i,z2i);      
      else
      dist=integral(@hubble,z1,z2(i),'AbsTol',1e-6);
      dist=light.*dist;
      end
      dcomov(1,i)=dist;
      dangle(1,i)=dist./(1.+z2(i));
  end
  
end

end    



