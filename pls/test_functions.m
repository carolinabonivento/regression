%x=data(1:3,1:5)
%y=data(4:6,1:3)
sd=x'*y
ssd=sd*sd'
wd= eigs(ssd)
td=x*wd
cd=y'*(td/(td'*td))
pd=x'*(td/(td'*td))
