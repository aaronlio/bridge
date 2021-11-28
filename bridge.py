
import math as math
import random as random
from numpy.core.records import recarray
from tqdm import tqdm
import copy
import numpy as np

###CIV VARIABLES###
E = 4000
max_tensile = 30
max_compressive = 6
max_shear = 4
mu = 0.2 #poisson's ratio
max_shear_cement = 2
pi = math.pi

###CONSTRAINTS###
mandatory_length = 1280
total_matboard = 813*1016 #826008

class Bridge:
    def __init__(self, height, length, glue_tab_width, num_top_flange_layers, num_bottom_flange_layers, num_web_layers, web_dist):
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
        self.top_flange_thickness = self.num_top_flange_layers*self.paper_thickness
        self.bottom_flange_thickness = self.num_bottom_flange_layers*self.paper_thickness
        self.web_thickness = num_web_layers*self.paper_thickness

    def SFD_train(self):
        wheel_positions = [0, 176, 340, 516, 680, 856]
        support_a_position = 15
        support_b_position = 1075
        reaction_forces = [[],[]]
        for i in range(1280-wheel_positions[5]):
            reaction_b = 0
            reaction_a = 0
            relative_wheel_positions = list(map(lambda x: x - support_a_position, wheel_positions))
            reaction_b = (sum(relative_wheel_positions)*(400/6))/(support_b_position-support_a_position)
            reaction_a = 400-reaction_b
            
            wheel_positions = list(map(lambda x: x + 1, wheel_positions))
            reaction_forces[0].append(reaction_a)
            reaction_forces[1].append(reaction_b)
            print(wheel_positions)
        return reaction_forces
    
    def point_SFD(self):
        position = np.linspace(0, self.length, 1)
        self.point_shear = np.zeros((1, self.length))
        self.point_shear[0][15:564] = .302
        self.point_shear[0][565:1059] = -.698
        self.point_shear[0][1060:1264] = 1

    def train_BMD(self):
        mid_wheel_positions = [132, 308, 472, 648, 812, 988] 
        end_wheel_positions = [424, 600, 764, 940, 1104, 1280]
        support_a_position = 15
        support_b_position = 1075
        self.train_bmd = np.zeros((1, self.length))
        
        reaction_forces = self.SFD_train()
        reaction_a_mid_wheel, reaction_b_mid_wheel = reaction_forces[0][mid_wheel_positions[0]-support_a_position], reaction_forces[1][mid_wheel_positions[0]-support_a_position]
        reaction_a_end, reaction_b_end = reaction_forces[0][end_wheel_positions[0]-support_a_position], reaction_forces[1][end_wheel_positions[0]-support_a_position]

        ind_shear_values_at_mid = [0, reaction_a_mid_wheel, -400/6, -400/6,-400/6,-400/6,-400/6,-400/6, reaction_b_mid_wheel]
    
        
        moment_mid = [(15,0), (102.5, 20400), (278.5,43860), (442.5, 54783 ), (618.5, 54783), (782.5, 43860 ), (958.5, 20400)]

        positions_train_at_end = [0, 15, 424, 600, 764, 940, 1075, 1104, 1280]
        shear_values_train_at_end = [0, reaction_a_end, -400/6,-400/6,-400/6,-400/6, reaction_b_end, -400/6,-400/6]
        moment_end = [(0,0), (15, 0), (424, 35385), (600, 39457), (764, 32319), (940,20000), (1075, -13603), (1104, -11736), (1280, 0)]
        
        #return self.bmd
        
    def ybar(self):
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        length = self.length
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness

        top_flange_a = flange_width*(top_flange_thickness)
        bottom_flange_a = web_dist * (bottom_flange_thickness)
        web_a = 2*((web_thickness)*(height-top_flange_thickness-bottom_flange_thickness-self.glue_tab_thickness))
        glue_tabs_a = (self.glue_tab_thickness*self.glue_tab_width)*2

        top_flange_Y = height - (top_flange_thickness/2)
        bottom_flange_Y = bottom_flange_thickness/2
        web_Y = (height - top_flange_thickness-self.glue_tab_thickness-bottom_flange_thickness)/2 + bottom_flange_thickness
        glue_tabs_y = height-top_flange_thickness-(self.glue_tab_thickness/2)
        
        top_flange_aY = top_flange_a*top_flange_Y
        bottom_flange_aY = bottom_flange_a*bottom_flange_Y
        web_aY = web_a * web_Y
        glue_tabs_ay = glue_tabs_a*glue_tabs_y
        
        total_ay = top_flange_aY + bottom_flange_aY + web_aY + glue_tabs_ay
        ybar = total_ay/(top_flange_a + bottom_flange_a + web_a + glue_tabs_a)
        
        return ybar

    def get_I_A(self):#Get the I for the A section! (It's still pi shaped)
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        length = self.length
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness

        top_flange_a = flange_width*(top_flange_thickness)
        bottom_flange_a = web_dist * (bottom_flange_thickness)
        web_a = 2*((web_thickness)*(height-top_flange_thickness-bottom_flange_thickness-self.glue_tab_thickness))
        glue_tabs_a = (self.glue_tab_thickness*self.glue_tab_width)*2

        top_flange_Y = height - (top_flange_thickness/2)
        bottom_flange_Y = bottom_flange_thickness/2
        web_Y = (height - top_flange_thickness-self.glue_tab_thickness-bottom_flange_thickness)/2 + bottom_flange_thickness
        glue_tabs_y = height-top_flange_thickness-(self.glue_tab_thickness/2)

        yb = self.ybar()
        top_flange_d = abs(top_flange_Y-yb)
        bottom_flange_d = abs(yb-bottom_flange_Y)
        web_d = abs(web_Y-yb)
        glue_tabs_d = abs(glue_tabs_y-yb)
        
        top_flange_ay2 = top_flange_a*(top_flange_d**2)
        bottom_flange_ay2 = bottom_flange_a*(bottom_flange_d**2)
        web_ay2 = web_a*(web_d**2)
        glue_tabsay2 = glue_tabs_a*(glue_tabs_d**2)
        
        web_I = ((web_thickness)*((height - top_flange_thickness - bottom_flange_thickness - self.glue_tab_thickness)**3))/12
        top_flange_I = (self.flange_width*(top_flange_thickness**3))/12
        bottom_flange_I = (self.web_dist*(bottom_flange_thickness**3))/12
        glue_tabs_I = (self.glue_tab_width *(self.glue_tab_thickness**3))/12
        
        sum_ay2 = top_flange_ay2 + (2*web_ay2) + bottom_flange_ay2 + (2*glue_tabsay2)
        sum_I = web_I + bottom_flange_I + top_flange_I + glue_tabs_I
        
        return sum_ay2+sum_I
		
    def get_max_P_flexural_A(self): #Calculates the maximum P the bridge can handle without flexural failure.
        height = self.height
        flange_width = self.flange_width
        web_dist = self.web_dist
        length = self.length
        top_flange_thickness = self.top_flange_thickness
        bottom_flange_thickness = self.bottom_flange_thickness
        web_thickness = self.web_thickness
        
        I = self.get_I_A()
        y = self.ybar()

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
        
        
        I = self.get_I_A()
        y = self.ybar()
        
        Q = ((web_thickness*2)*(y-bottom_flange_thickness)*(y-abs(y-bottom_flange_thickness)/2)) + (bottom_flange_thickness*web_dist)*abs(y-(bottom_flange_thickness/2))
        P5 = (max_shear*I*(2*web_thickness))/Q

        self.Qglue = (top_flange_thickness*flange_width)*(height - (top_flange_thickness/2) - y)
        Qglue = self.Qglue
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
        sigma_crit_case_1 = 3.455
        sigma_crit_case_2 = 23.5
        sigma_crit_case_3 = 31.864
        sigma_crit_case_4 = 3.45
        I = self.I
        Q = self.Q
        b = self.paper_thickness
        Qglue = self.Qglue
        
        # Scenario 1 - Train is between supports
        M_plus_max = 54830
        ytop = self.ytop
        ybot = self.ybot
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
        print(I)
        shearGlueFOS2 = (4/((V1*Qglue)/I*(2*(self.glue_tab_width+self.bottom_flange_thickness))))

        bucklingFOSCase1_2 = sigma_crit_case_1/((M_plus_max*ytop)/I)
        bucklingFOSCase2_2 = sigma_crit_case_2/((M_plus_max*ytop)/I)
        bucklingFOSCase3_2 = sigma_crit_case_3/((M_plus_max*ytop)/I)

        bucklingFOSCase4_2 = sigma_crit_case_4/((M_neg_max*ybot)/I)

        FOSes = [FOSc, FOSt, shearFOS, shearGlueFOS,bucklingFOSCase1, bucklingFOSCase2, bucklingFOSCase3, bucklingFOSCase4, 
        FOSc2a, FOSt2a, FOSc2b, FOSt2b, shearFOS2, shearGlueFOS2, bucklingFOSCase1_2, bucklingFOSCase2_2, bucklingFOSCase3_2, bucklingFOSCase4_2]

        return FOSes


    def report(self):
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
            "Buckling Failure of Top of Web": self.get_buckling_failure_A()[3],
            "Buckling Failure of Top Flange at ends": self.get_buckling_failure_A()[4],
            "Buckling Failure of bottom flange": self.get_buckling_failure_A()[5],
            "Shear Buckling Failure of Top of Web with a = 520": self.get_buckling_failure_A()[0],
            "Shear Buckling Failure of Top of Web with a = 480": self.get_buckling_failure_A()[1]}
            
        FOSes = self.FOS()
        print(FOSes)




b1 = Bridge(75, 1280, 11.27, 1, 1, 1, 80)
b1.report()