%Values 

nm=1.333;
wavelength = 532;
k=2*pi*nm/wavelength;
    
fun = @(R, theta) (3./((2*k).*sin((theta-pi/2)/2)*R).^3.*sin((2*k*sin((theta-pi/2)/2))*R)-(2*k*sin((theta-pi/2)/2)*R).*cos((2*k*sin((theta-pi/2)/2))*R)).^2;


Evaluate the integral from x=0 to x=Inf. 
radius_List = 1:1:300;
form_factor_list = zeros([length(radius_List), 1]);
for i=1:length(radius_List)
    R= radius_List(i);
    fun = @(theta) (3./((2*k).*sin((theta-pi/2)/2)*R).^3.*(sin((2*k*sin((theta-pi/2)/2))*R)-(2*k*sin((theta-pi/2)/2)*R).*cos((2*k*sin((theta-pi/2)/2))*R))).^2;
    form_factor_list(i) = integral(fun, -pi/10, pi/10);
end
save('myArray_integralSquared.mat', 'form_factor_list')  