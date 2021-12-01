from itertools import product
import math as math
import random as random
from numpy.core.records import recarray
from tqdm import tqdm
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy

###Important Constants###
E = 4000
max_tensile = 30
max_compressive = 6
max_shear = 4
mu = 0.2 #poisson's ratio
max_shear_cement = 2
pi = math.pi
global failure_load
###CONSTRAINTS###
mandatory_length = 1280
split_point = 1016 #When moment is 0 under train loading B
total_matboard = 813*1016 #826008

#creating the bridge object class
class Bridge:
    def __init__(self, height, length, glue_tab_width, num_top_flange_layers, num_bottom_flange_layers, num_web_layers, web_dist, dia_dist, dia_num):
        """
        The constructor of the bridge object, which contains all the parameter, or characteristic showing above.
        """
        #Important values
        self.paper_thickness = 1.27
        self.height = height
        self.length = length
        self.flange_width = 100
        self.num_top_flange_layers = num_top_flange_layers
        self.num_bottom_flange_layers = num_bottom_flange_layers
        self.num_web_layers = num_web_layers
        self.glue_tab_width = glue_tab_width
        self.glue_tab_thickness = self.paper_thickness
        self.web_dist = web_dist
        self.dia_num=dia_num
        self.top_flange_thickness = self.num_top_flange_layers*self.paper_thickness
        self.bottom_flange_thickness = self.num_bottom_flange_layers*self.paper_thickness
        self.web_thickness = num_web_layers*self.paper_thickness
        self.dia_dist = dia_dist
        self.I = self.get_I_A()

    # Calculates the reaction forces under any train loading conditions
    def reaction_forces(self, wheel_positions):
        """
        The function will get the position of the train wheels, and return the reaction force as a list of list
        """
        support_a_position = 15
        support_b_position = 1075
        reaction_forces = [0,0]

        # Get the wheel positions relative to support A; these values can be used in moment calculations
        relative_wheel_positions = list(map(lambda x: x - support_a_position, wheel_positions))
        #Sum of moments @ A = 0, solve for B
        reaction_b = sum(relative_wheel_positions)*(400/6)/1060
        #Sum in Y = 0, solve for a
        reaction_a = 400-reaction_b
            
        reaction_forces[0] = reaction_a
        reaction_forces[1] = reaction_b

        return reaction_forces
        
    def ybar(self):
        """
        The function calculates the central axis value on y using its dimension
        """
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        length = self.length
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness
        #getting key values
        top_flange_a = flange_width*(top_flange_thickness)
        bottom_flange_a = web_dist * (bottom_flange_thickness)
        web_a = 2*((web_thickness)*(height-top_flange_thickness-bottom_flange_thickness-self.glue_tab_thickness))
        glue_tabs_a = (self.glue_tab_thickness*self.glue_tab_width)*2
        #area of all the different shapes
        top_flange_Y = height - (top_flange_thickness/2)
        bottom_flange_Y = bottom_flange_thickness/2
        web_Y = (height - top_flange_thickness-self.glue_tab_thickness-bottom_flange_thickness)/2 + bottom_flange_thickness
        glue_tabs_y = height-top_flange_thickness-(self.glue_tab_thickness/2)
        #the Y value from the bottom of each component
        top_flange_aY = top_flange_a*top_flange_Y
        bottom_flange_aY = bottom_flange_a*bottom_flange_Y
        web_aY = web_a * web_Y
        glue_tabs_ay = glue_tabs_a*glue_tabs_y
        #times up the area with the Y value
        total_ay = top_flange_aY + bottom_flange_aY + web_aY + glue_tabs_ay
        ybar = total_ay/(top_flange_a + bottom_flange_a + web_a + glue_tabs_a)
        #calcuate the actual ybar
        return ybar

    def get_I_A(self):#Get the I for the A section! (It's still pi shaped)
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        length = self.length
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness
        #assigning the variables
        top_flange_a = flange_width*(top_flange_thickness)
        bottom_flange_a = web_dist * (bottom_flange_thickness)
        web_a = 2*((web_thickness)*(height-top_flange_thickness-bottom_flange_thickness-self.glue_tab_thickness))
        glue_tabs_a = (self.glue_tab_thickness*self.glue_tab_width)*2
        #calcuating the area
        top_flange_Y = height - (top_flange_thickness/2)
        bottom_flange_Y = bottom_flange_thickness/2
        web_Y = (height - top_flange_thickness-self.glue_tab_thickness-bottom_flange_thickness)/2 + bottom_flange_thickness
        glue_tabs_y = height-top_flange_thickness-(self.glue_tab_thickness/2)
        #calculating the Y value
        yb = self.ybar()
        top_flange_d = abs(top_flange_Y-yb)
        bottom_flange_d = abs(yb-bottom_flange_Y)
        web_d = abs(web_Y-yb)
        glue_tabs_d = abs(glue_tabs_y-yb)
        #calculating the distance from the height to the y bar
        top_flange_ay2 = top_flange_a*(top_flange_d**2)
        bottom_flange_ay2 = bottom_flange_a*(bottom_flange_d**2)
        web_ay2 = web_a*(web_d**2)
        glue_tabsay2 = glue_tabs_a*(glue_tabs_d**2)
        #value of area times d^2
        web_I = ((web_thickness)*((height - top_flange_thickness - bottom_flange_thickness - self.glue_tab_thickness)**3))/12
        top_flange_I = (self.flange_width*(top_flange_thickness**3))/12
        bottom_flange_I = (self.web_dist*(bottom_flange_thickness**3))/12
        glue_tabs_I = (self.glue_tab_width *(self.glue_tab_thickness**3))/12
        #calculate the I value for each component
        sum_ay2 = top_flange_ay2 + (2*web_ay2) + bottom_flange_ay2 + (2*glue_tabsay2)
        sum_I = web_I + bottom_flange_I + top_flange_I + glue_tabs_I
        #sum up all the values
        return sum_ay2+sum_I
		
    def get_max_P_flexural_A(self): #Calculates the maximum P the bridge can handle without flexural failure.
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        length = self.length
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness
        #assigning values
        self.I = self.get_I_A()
        I = self.I
        y = self.ybar()
        #get the I and y bar value
        self.ytop = height - y
        self.ybot = y
        ytop = self.ytop
        ybot = self.ybot
         #M+max = 166P
         #M-max = 190P

		#Calculating P for Tension failure, M+ at bot, M- at top
        M = (max_tensile*I)/ybot #maximum moment that this section can endure...
        P = M/166
        P1 = P

        M = (max_tensile*I)/ytop
        P = M/190
        P2 = P

		#Calculating P for Compressive failure
        M = (max_compressive*I)/(ybot)
        P = M/190
        P3 = P

        M = (max_compressive*I)/(ytop)
        P = M/166
        P4 = P
        
        return [P1, P2, P3, P4]

    def get_max_P_shear_A(self):
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        length = self.length
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness
        
        #assigning the values
        self.I = self.get_I_A()
        I = self.I
        y = self.ybar()
        #getting value of I and y bar
        Q = ((web_thickness*2)*(y-bottom_flange_thickness)*(y-abs(y-bottom_flange_thickness)/2)) + (bottom_flange_thickness*web_dist)*abs(y-(bottom_flange_thickness/2))
        P5 = (max_shear*I*(2*web_thickness))/Q
        
        Qglue = (top_flange_thickness*flange_width)*(height - (top_flange_thickness/2) - y)
        P6 = (max_shear_cement*I*(self.glue_tab_width))/Qglue

        return [P5,P6]

    def get_buckling_failure_A(self):
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        length = self.length
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness
        #assigning all values
        self.I = self.get_I_A()
        I= self.I
        y = self.ybar()

        self.Q = ((web_thickness*2)*(y-bottom_flange_thickness)*(y-abs(y-bottom_flange_thickness)/2)) + (bottom_flange_thickness*web_dist)*abs(y-(bottom_flange_thickness/2))
        Q = self.Q
		#Case 1: Compressive flange at top
        sig_comp_crit1 = ((4*(pi**2)*E)/(12*(1-mu**2)))*((top_flange_thickness/web_dist)**2)
        P7 = (sig_comp_crit1*I)/(166*self.ytop)

		#Case 2: Flexural compression @ top of web
        sig_comp_crit2 = ((6*(pi**2)*E)/(12*(1-mu**2)))*(( web_thickness /(height-y-top_flange_thickness-self.glue_tab_thickness))**2)
        P8 = (sig_comp_crit2*I)/(166*self.ytop)

        #Case 2b: Flexural compression @ bottom of web
        sig_comp_crit4 = ((6*(pi**2)*E)/(12*(1-mu**2)))*(( web_thickness /(y-bottom_flange_thickness))**2)
        P9 = (sig_comp_crit4*I)/(166*self.ytop)

        #Case 3: Compressive buckling at the edge of the thing.
        b = (flange_width - web_dist)/2
        sig_comp_crit3 = ((0.425*(pi**2)*E)/(12*(1-mu**2)))*((top_flange_thickness/((b)))**2)
        P10 = (sig_comp_crit3*I)/(166*self.ytop)

        #Case 4: Compressive buckling at base
        P11 = (sig_comp_crit1*I)/(190*self.ybot)

		#shear buckling the top of the web
        a1 = 520

        a2 = 480
        V2 = 1
        
        self.Tau_crit1 = ((5*(pi**2)*E)/(12*(1-(mu**2))))*(((web_thickness/(height-top_flange_thickness-bottom_flange_thickness))**2) + ((web_thickness/a1)**2))
        Tau_crit1 = self.Tau_crit1
        P12 = (Tau_crit1*I*2*web_thickness)/(Q*.698)
        self.Tau_crit2 = ((5*(pi**2)*E)/(12*(1-(mu**2))))*(((web_thickness/(height-top_flange_thickness-bottom_flange_thickness))**2) + ((web_thickness/a2)**2))
        Tau_crit2 = self.Tau_crit2
        V2 = (Tau_crit2*I*(2*web_thickness))/Q
        P13 = V2

        P12 = max(P12, P13)
        
        return [P7, P8, P9, P10, P11, P12, P13]

    def get_max_load_A(self):
        return min(self.get_max_P_shear_A(), min(self.get_max_P_flexural_A()), min(self.get_buckling_failure_A()))

    # get FOS's, use calculations
    def FOS(self):
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        length = self.length
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness
        y = self.ybar()

        ytop = height - y
        ybot = y

        sigma_crit_case_1 = 3.455
        sigma_crit_case_2 = 23.5
        sigma_crit_case_3 = 31.864
        sigma_crit_case_4 = 3.45
        Q = ((web_thickness*2)*(y-bottom_flange_thickness)*(y-abs(y-bottom_flange_thickness)/2)) + (bottom_flange_thickness*web_dist)*abs(y-(bottom_flange_thickness/2))
        I = self.I
        b = self.paper_thickness
        Qglue = (top_flange_thickness*flange_width)*(height - (top_flange_thickness/2) - y)
        
        # Scenario 1 - Train is between supports
        M_plus_max = 54830
        FOSc = 6/((54830*ytop)/I)
        FOSt = 30/((54830*ybot)/I)
        shearFOS = (4/((200*Q)/(I*(2*b))))
        shearGlueFOS = (2*(2*I*(self.glue_tab_width))/(200*Qglue))
        bucklingFOSCase1 = sigma_crit_case_1/((54830*ytop)/I)
        bucklingFOSCase2 = sigma_crit_case_2 / ((54830*ytop)/I)
        bucklingFOSCase3 = sigma_crit_case_3 / ((54830*ytop)/I)
        bucklingFOSCase4 = sigma_crit_case_4 / ((54830*ytop)/I)

        # Scenario 2 - Train is on support B
        M_plus_max = 39457
        M_neg_max = 13603
        V1 = 176.85

        FOSc2a = (6/((M_plus_max*ytop)/I))
        FOSt2a = (30/((M_plus_max*ybot)/I))
        
        FOSc2b = (30/((M_neg_max*ytop)/I))
        FOSt2b = (6/((M_neg_max*ybot)/I))

        shearFOS2 = 4/((V1*Q)/(I*2*b))
        shearGlueFOS2 = (4/((V1*Qglue)/(I*(2*(self.glue_tab_width)))))

        bucklingFOSCase1_2 = sigma_crit_case_1/((M_plus_max*ytop)/I)
        bucklingFOSCase2_2 = sigma_crit_case_2/((M_plus_max*ytop)/I)
        bucklingFOSCase3_2 = sigma_crit_case_3/((M_plus_max*ytop)/I)
        bucklingFOSCase4_2 = sigma_crit_case_4/((M_neg_max*ybot)/I)

        FOSes = [FOSc, FOSt, shearFOS, shearGlueFOS,bucklingFOSCase1, bucklingFOSCase2, bucklingFOSCase3, bucklingFOSCase4, 
        FOSc2a, FOSt2a, FOSc2b, FOSt2b, shearFOS2, shearGlueFOS2, bucklingFOSCase1_2, bucklingFOSCase2_2, bucklingFOSCase3_2, bucklingFOSCase4_2]
        
        
        return min(FOSes)
    
    def deadLoad(self):
        failure_modes = {
            "Tension failure at bottom": self.get_max_P_flexural_A()[0], 
            "Tension failure at top": self.get_max_P_flexural_A()[1], 
            "Compressive failure at bottom": self.get_max_P_flexural_A()[2], 
            "Compressive failure at top": self.get_max_P_flexural_A()[3],
            "Matboard Shear Failure": self.get_max_P_shear_A()[0],
            "Glue Shear Failure": self.get_max_P_shear_A()[1],
            "Buckling Failure of Top Flange at Mid": self.get_buckling_failure_A()[0],
            "Buckling Failure of Top of Web": self.get_buckling_failure_A()[1],
            "Buckling Failure of Bottom of Web": self.get_buckling_failure_A()[2],
            "Buckling Failure of Top Flange at ends": self.get_buckling_failure_A()[3],
            "Buckling Failure of bottom flange": self.get_buckling_failure_A()[4],
            "Shear Buckling Failure of Top of Web with a = 520": self.get_buckling_failure_A()[5],
            "Shear Buckling Failure of Top of Web with a = 480": self.get_buckling_failure_A()[6]}
        failure_load = min(zip(failure_modes.values(), failure_modes.keys()))[1]
        return failure_modes[failure_load]
    
    def deadTrain(self):
        return self.FOS()
        
    def report(self):
        global failure_load
        #the following dictionary contains all types of failing ways as well as
        #its correesponding failing load
        failure_modes = {
            "Tension failure at bottom": self.get_max_P_flexural_A()[0], 
            "Tension failure at top": self.get_max_P_flexural_A()[1], 
            "Compressive failure at bottom": self.get_max_P_flexural_A()[2], 
            "Compressive failure at top": self.get_max_P_flexural_A()[3],
            "Matboard Shear Failure": self.get_max_P_shear_A()[0],
            "Glue Shear Failure": self.get_max_P_shear_A()[1],
            "Buckling Failure of Top Flange at Mid": self.get_buckling_failure_A()[0],
            "Buckling Failure of Top of Web": self.get_buckling_failure_A()[1],
            "Buckling Failure of Bottom of Web": self.get_buckling_failure_A()[2],
            "Buckling Failure of Top Flange at ends": self.get_buckling_failure_A()[3],
            "Buckling Failure of bottom flange": self.get_buckling_failure_A()[4],
            "Shear Buckling Failure of Top of Web with a = 520": self.get_buckling_failure_A()[5],
            "Shear Buckling Failure of Top of Web with a = 480": self.get_buckling_failure_A()[6]}
        
        #for the point load test
        failure_load = min(zip(failure_modes.values(), failure_modes.keys()))[1]
        print(f"The way to fail the bridge is: {failure_load}")
        print(f"The maximun load that will fail the bridge is: {failure_modes[failure_load]}")
        #for the train test
        FOSes = self.FOS()
        print(f"Pass the train test: {FOSes}")
    
    def get_amount_paper(self):
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness
        length = self.length
        dia_dist = self.dia_dist
        dia_num=self.dia_num
        num_flange_layers_top = self.num_top_flange_layers
        num_flange_layers_bottom = self.num_bottom_flange_layers
        num_web_layers = self.num_web_layers
        glue_tab_width=self.glue_tab_width
        paper_used=0
        top_layer_A = (flange_width*length*num_flange_layers_top)
        each_side_layer_A=(num_web_layers*(height-top_flange_thickness-bottom_flange_thickness+2*glue_tab_width)*length)
        #with glue tab!!
        bottom_layer_A=web_dist*length*num_flange_layers_bottom
        paper_used+=(top_layer_A+2*each_side_layer_A+bottom_layer_A)
        num_diaphragms = (length/dia_dist)+5
        area_dia = dia_num*(height-top_flange_thickness-bottom_flange_thickness)*web_dist
        
        paper_used += area_dia
        
        return paper_used

     # Finds the shear forces from the train load and plots it alongside hand-calculated Vfail for design0
    # and the calculated Vfail for our design
    def SFD_train(self):
        global failure_load
        max_shear_force = []
        support_a_position = 15
        support_b_position = 1075
        max_single_shear_force = 0
        shear_with_position = {}
        shear_at_every_point = []
        max_shear = [0]
        for i in range(424):
            cur_shear_force = 0
            shear = []
            wheel_positions = [0+i, 176+i, 340+i, 516+i, 680+i, 856+i]
            wheel_positions_dict = {0+i: -400/6, 15: 0, 176+i: -400/6, 340+i: -400/6, 516+i: -400/6, 680+i: -400/6, 856+i: -400/6, 1075:0}
            
            reaction_forces = self.reaction_forces(wheel_positions)
            
            wheel_positions_dict[15] = reaction_forces[0] 
            wheel_positions_dict[1075] = reaction_forces[1]
            # print(wheel_positions_dict[15], wheel_positions_dict[1075])

            for j in range(self.length):
                if (j == 15 or j == 1075) and j in wheel_positions:
                    wheel_positions_dict[j] -= 400/6

                if j in wheel_positions_dict.keys():
                    cur_shear_force += wheel_positions_dict[j]
                    
                
                shear.append(cur_shear_force)
            
            shear_at_every_point.append(shear)
            
            if max(shear, key=abs) > max(max_shear, key=abs):
                max_shear = shear
        print(max(max_shear))

        ax = plt.subplot(1,1,1)

        x_axis = np.linspace(1,self.length, 1280)
        line1 = ax.plot(x_axis, max_shear, "-b", x_axis, [178]*1280, "-r", x_axis, [+648.350953242768] *1280, "-g", x_axis, [-178]*1280, "-r", x_axis, [-648.350953242768] *1280, "-g")
        plt.legend(['Shear Force Along Bridge', 'Design 0 Failure Load', 'Design 1 Failure Load'], loc='upper right')
        plt.title("SFD with Maximum Shear Force, Train at 16mm")
        plt.xlabel("Distance along bridge (mm)")
        plt.ylabel("Shear Force (N)")
        plt.show()


        return shear_at_every_point

    def BMD_train(self):
        shear = self.SFD_train()
        max_moment=0
        listOfMatMax=[]
        indexWhenMax=0
        for i in range(len(shear)):
            Ms=[]
            curM=0
            for j in range (len(shear[0])):
                curM+=shear[i][j]
                if(curM>max_moment): 
                    max_moment=curM
                    listOfMatMax=Ms
                    indexWhenMax=i
                Ms.append(curM)
        
        ax = plt.subplot(1,1,1)
        x_axis = np.linspace(1,self.length, 1280)
        line1=ax.plot(x_axis, listOfMatMax)
        #indexWhenMax is the i when max momemnt occured
        plt.show()

        

#self, height, length, glue_tab_width, num_top_flange_layers, num_bottom_flange_layers, num_web_layers, web_dist, dia_dist, dia_num
print("-------")
b1 = Bridge(75, 1280, 11.27, 2, 1, 1, 80, 550, 8)
b1.report()
print("-------")
b2=Bridge(99,1280,8,1, 3, 1,61, 520,8)
b2.report()
"""
lllll=[0,0,0,0,0, 0, 0]#min load, x, y, z,glueW, height, i
print(b1.get_amount_paper())
for x,y,z, glueW, height in product(range(1, 5), range(1, 5), range(1, 5), range(8,20), range(50,100)):
    for i in range (1,100):
        b_sample=Bridge(height, 1280, glueW, x,y,z, i, 520, 8)
        if (b_sample.deadLoad()>lllll[0] and b_sample.get_amount_paper()<813*1016*0.92 and b_sample.deadTrain()):
            lllll[0]=b_sample.deadLoad()
            lllll[1]=x
            lllll[2]=y
            lllll[3]=z
            lllll[4]=glueW
            lllll[5]=height
            lllll[6]=i
print (f"The size is: {lllll}")
"""