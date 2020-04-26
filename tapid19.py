import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from random import *
import time

#Copy your paragraph settings here :





#hard =  prop_conta_rate , step, step_conta , pot_infect, radius_meet, escape_rate

prop_conta_rate = np.array([[0.9,0.4,0.284],[0.09,0.4,0.5],[0.01,0.197,0.2],[0,0.003,0.016]])
step = [3,6,15,20]
step_conta = np.array([[0,0,0],[0,1,1],[0,1,2],[0,1,3]])
pot_infect = np.array([[0,0.7,0.7,0],[0,1,1,0],[0,1.2,0.6,0.6],[0,1.2,0.3,0]])
coef = 4 #Coefficient to build area for infect_choice
escape_rate = 0.01 #Rate of person leaving its area

#Setup a area
ymax = int(np.sqrt(population * 9 / 16))
xmax = int(ymax * 16/9)

#measures = closing_schools(meet_dicrease) , confinement(cluster_size, nb_meet_dicrease) , quarantine((xmin,ymin,xmax,ymax) , barrier_gestures(dim_pot_infect) , isolation_age(age,nb_meet_dicrease) , isolation_infected(nb_meet_dicrease)

closing_schools = [0,[0.1,0.4,0.8]]
confinement = [1,[5, 0.1,0.5]]
quarantine = [2,[10,10,40,40]]
barrier_gestures = [3,[0.7]]
isolation_age = [4,[2,0.1]]
isolation_infected = [5,[0.1]]

measures_list = [closing_schools,confinement,quarantine,barrier_gestures,isolation_age,isolation_infected]

#Size-based confinement settings
if confinement[1][0]==3 :
    length_cluster=2
    width_cluster=1
if confinement[1][0]==4 :
    length_cluster=2
    width_cluster=2
if confinement[1][0]==5 :
    length_cluster=5
    width_cluster=1
if confinement[1][0]==6 :
    length_cluster=3
    width_cluster=2

#Pick random infected if nb_infected given
def coord_infected_func(nb_infected) :
    coord_infected = []
    x = np.random.randint(0,xmax,size=(nb_infected))
    y = np.random.randint(0,ymax,size=(nb_infected))

    for k in range(nb_infected) :
        coord_infected.append((x[k],y[k]))

    return coord_infected


#Check if the user gived coord_infected
if len(coord_infected) == 0 :
    coord_infected = coord_infected_func(nb_infected)


#Average number of contamination by one infected for one meet
def nb_conta_av() :
    l = []
    res = 0
    for x in range(4) :
        for k in range(3) :
            res += prop_conta_rate[x][k]*prop_ages[k]*(nb_meet[k][0]+nb_meet[k][1]+nb_meet[k][2])
        res = res * (pot_infect[x][1] * 3 + pot_infect[x][2] * 9 + pot_infect[x][3] * 5)
        l.append(res)
        res = 0

    return (R0 / (l[0] + l[1] + l[2] + l[3]))

nb_conta_av = nb_conta_av()


# Table : [Age(0,1,2) ,(Healthy(0),Infected(1),Healed(2)), Condition(asymptomatic(0),benign(1),severe(2),fatal(3)), Symptoms(asymptomatic(0),minor(1),severe(2),fatal(3)), Infection_date]

#Condition is defined at the infection time and predefines the evolution of symptomes of the infected


def initialization_table() :
    table = np.zeros((xmax,ymax,7))
    for k in range(xmax) :
        for l in range(ymax) :
            rd = random()
            if rd < prop_ages[0] :
                table[k][l][0] = 0
            if rd >= prop_ages[0] and rd < prop_ages[0] + prop_ages[1] :
                table[k][l][0] = 1
            if prop_ages[0] + prop_ages[1] <= rd :
                table[k][l][0] = 2
    #Initial infected choice
    for m in coord_infected_func(nb_infected) :
        x,y = m
        rd = random()
        age = int(table[x][y][0]) #age
        table[x][y][1] = 1 #infected
        #Condition
        if rd < prop_conta_rate[0][age] :
            table[x][y][2] = 0
        if rd > prop_conta_rate[0][age] and prop_conta_rate[1][age] + prop_conta_rate[0][age] :
            table[x][y][2] = 1
        if rd > prop_conta_rate[0][age] + prop_conta_rate[1][age] and rd < prop_conta_rate[0][age] + prop_conta_rate[1][age] + prop_conta_rate[2][age] :
            table[x][y][2] = 2
        if rd > prop_conta_rate[0][age] + prop_conta_rate[1][age] + prop_conta_rate[2][age] :
            table[x][y][2] = 3
    return table

def day_measures_func(day) :

    #Return a table with 1 if the corresponding measure is up and 0 otherwise

    day_measures = np.zeros(6)
    for k in range(len(scenario)) :
        begin_date = scenario[k][1]
        ending_date = scenario[k][2]
        if day >= begin_date and day <= ending_date :
            day_measures[scenario[k][0]] = 1

    #Giving priorities for some measures on others if done at the same time

    #School closing and Isolation of age 0 gives priority to Isolation of age 0
    if day_measures[4] == 1 and isolation_age[1][0] == 0 and day_measures[0] == 1 :
        day_measures[0] = 0

    #Confinement and Isolement age gives priority to confinement
    if day_measures[1] == 1 and day_measures[4] == 1 :
        day_measures[4] = 0
    return day_measures




#Return a symptome from a time of infection
def symptome_func(time_infect) :
    symptome = 0
    if time_infect < step[1] and time_infect > step[0] :
        symptome = 1
    if time_infect >= step[1] and time_infect < step[2] :
        symptome = 2
    if time_infect >= step[2] :
        symptome = 3
    return int(symptome)

#Make the choice of new contaminated by creating an area around the infected
#The area is larger than the number of met people to create a better simulation
#All changes due to measures are implemented in this function
def infect_choice(table, nb_conta_av,date) :
    infected_list = []
    meet = []
    day_measures = day_measures_func(date)

    #Setup coefficient for measures
    coef_meet = []
    coef_isol_age = 1
    age_isol = 0
    coef_isol_infect = 1

    #Closing schools
    if day_measures[0] == 1 :
        coef_meet = [closing_schools[1][0], closing_schools[1][1] ,closing_schools[1][2]]
    else :
        coef_meet = [1,1,1]
    if day_measures[4] == 1 :
        coef_isol_age = isolation_age[1][1]
        age_isol = isolation_age[1][0]
    else :
        coef_isol_age = 1

    #Barrier gestures
    coef_pot = []
    if day_measures[3] == 1 :
        coef_pot = barrier_gestures[1][0]
    else :
        coef_pot = 1

    #Confinement
    if day_measures[1] == 1 :
        radius_dicrease = confinement[1][2] #Coefficient to reduce number of potential meet(so area radius) meet when confinement is up
        coef_meet_conf = confinement[1][1]
    else :
        radius_dicrease=1
        coef_meet_conf=1

    #Quarantine
    qua = False #Quarantine not up
    if day_measures[2] == 1 :
        zone = quarantine[1]
        qua = True

    #Isolement infected
    if day_measures[5] == 1 :
        coef_isol_infect = isolation_infected[1][0]
    else :
        coef_isol_age = 1

    for k in range(xmax) :
        for l in range(ymax) :
            if table[k][l][1] == 1 and table[k][l][2] != 3 : #Alive infected
                age = int(table[k][l][0])
                time_infect = date - table[k][l][4] #Time of infection
                symptome = symptome_func(time_infect)
                in_zone = False # Infected not in quarantine zone
                if qua and k > zone[0] and k < zone[2] and l > zone[1] and l < zone[3] :
                    in_zone = True

                #Number of meet depending age and closing schools
                if age == 0 :
                    meet = [nb_meet[age][0]*coef_meet[0],nb_meet[age][1]*coef_meet[1],nb_meet[age][2]*coef_meet[2]]
                else :
                    meet = [nb_meet[age][0],nb_meet[age][1],nb_meet[age][2]]

                #Number of meet depending age of isolation age
                if age == age_isol :
                    meet = [meet[0]*coef_isol_age,meet[1]*coef_isol_age,meet[2]*coef_isol_age]
                else :
                    meet = [meet[0],meet[1],meet[2]]

                #Number of meet depending isolement infected measure
                if table[k][l][2] == 1 or table[k][l][2] == 2 :
                    meet = [meet[0]*coef_isol_infect,meet[1]*coef_isol_infect,meet[2]*coef_isol_infect]
                else :
                    meet = [meet[0],meet[1],meet[2]]

                #Number of person that the infected could potentially meet with confinment or not
                nb_pe = max(meet[0] / prop_ages[0],meet[1] / prop_ages[1],meet[2] / prop_ages[2])* coef*radius_dicrease

                side = int(np.sqrt(nb_pe) + 1) #Length of square side (for area)

                #Setup area around infected if it do not escape and random if it escape
                rd = random()
                if rd <= escape_rate :
                    a = randint(0,xmax)
                    b = randint(0,ymax)
                if rd > escape_rate :
                    a = k
                    b = l

                #Area coordonates
                zxmin = int(max(0,a - side / 2 ))
                zymin = int(max(0,b - side / 2))
                zxmax = int(min(xmax,a + side / 2))
                zymax = int(min(ymax,b + side / 2))

                #Probability to meet someone depending age
                proba_meet = [meet[0]*coef_meet_conf / (nb_pe * prop_ages[0]), meet[1]*coef_meet_conf / (nb_pe * prop_ages[1]),meet[2]*coef_meet_conf / (nb_pe * prop_ages[2])]

                #Probability of contamination depending symptomes with or without barrrier gestures
                proba_conta = nb_conta_av * pot_infect[int(table[k][l][2])][symptome]*coef_pot

                #Browse of the area, choice of infected
                for x in range(zxmin,zxmax) :
                    for y in range(zymin,zymax) :

                        #Because of quarantine measure
                        if qua and in_zone and x > zone[0] and x < zone[2] and y > zone[1] and y < zone[3] :
                            rd = random()
                            if rd < proba_meet[int(table[x][y][0])]*proba_conta :
                                infected_list.append([x,y])
                        if qua and not in_zone and not (x > zone[0] and x < zone[2] and y > zone[1] and y < zone[3]) :
                            rd = random()
                            if rd < proba_meet[int(table[x][y][0])]*proba_conta :
                                infected_list.append([x,y])
                        if not qua :
                            rd = random()
                            if rd < proba_meet[int(table[x][y][0])]*proba_conta :
                                infected_list.append([x,y])

                #Because of confinement
                if day_measures[1] == 1 :
                    cluster_coordx = int(k / length_cluster) * length_cluster
                    cluster_coordy = int(k / width_cluster) * width_cluster

                    #Browse cluster, choice of infected
                    for x in range(cluster_coordx, min(cluster_coordx + length_cluster - 1, xmax)) :
                        for y in range(cluster_coordy, min(cluster_coordy + width_cluster - 1, ymax)) :
                            rd = random()
                            if rd < proba_conta :
                                infected_list.append([x,y])
    return infected_list





#Update table for a given date by refering to condition of the infected
def evolve(table,date) :
    death_prob = 1/(step[2] - step[1])
    for k in range(xmax) :
        for l in range(ymax) :
            if table[k][l][1] == 1 :
                infect_start = table[k][l][4] #Date of infection
                duration = date - infect_start

                #Conditon : death
                if table[k][l][2] == 3 :
                    if duration >= step[0]  and duration < step[1]:
                        table[k][l][3] = 1 #Symptom : minor
                    if duration >= step[1] and  table[k][l][3] != 3 :
                        rd = random()
                        if rd < death_prob or duration == step[2] :
                            table[k][l][3] = 3 #Symptome : death
                        else :
                            table[k][l][3] = 2 #Symptom : severe

                #Condition : severe
                if table[k][l][2] == 2 :
                    if duration >= step[0]  and duration < step[1] :
                        table[k][l][3] = 1 #Symptom : minor
                    if duration >= step[1] and duration < step[3] :
                        table[k][l][3] = 2 #Symptom : severe
                    if duration == step[3] :
                        table[k][l][3] = 0 #Symptom : asymptomatique
                        table[k][l][1] = 2 #Healed

                #Condition : benign
                if table[k][l][2] == 1 :
                    if duration >= step[0]  and duration < step[1] :
                        table[k][l][3] = 1 #Symptom : minor
                    if duration >= step[1] and duration < step[2] :
                        table[k][l][3] = 1 #Symptom : minor
                    if duration == step[2] :
                        table[k][l][3] = 0 #Symptom : asymptomatique
                        table[k][l][1] = 2 #Healed

                #Condition : asymptomatic
                if table[k][l][2] == 0 :
                    if duration == step[2] :
                        table[k][l][3] = 0 #Symptom : asymptomatique
                        table[k][l][1] = 2 #Healed
    return(table)

#Update the table implementing infected selected by infect_choice and chosing a condition
def infect_update(table,infected_list,date) :
    for k in infected_list :
        x = k[0]
        y = k[1]
    #Do not modify already infected
        if table[x][y][1] == 0 :
            table[x][y][4] = date #Date of infection
            table[x][y][1] = 1 #Infected
            age = int(table[x][y][0]) #Age
            rd = random()

            #Choice of condition
            if rd < prop_conta_rate[0][age] :
                table[x][y][2] = 0  #Condition : asymptomatic

            if rd > prop_conta_rate[0][age] and prop_conta_rate[1][age] + prop_conta_rate[0][age] :
                table[x][y][2] = 1 #Condition : benign

            if rd > prop_conta_rate[0][age] + prop_conta_rate[1][age] and rd < prop_conta_rate[0][age] + prop_conta_rate[1][age] + prop_conta_rate[2][age] :
                table[x][y][2] = 2 #Condition : severe

            if rd > prop_conta_rate[0][age] + prop_conta_rate[1][age] + prop_conta_rate[2][age] :
                table[x][y][2] = 3 #Conditon : death

    return table


#Select data we want for animation
def history_ani(initialization,time) :
    history_anim = np.zeros((time,xmax,ymax))
    history = initialization

    for k in range(time) :
        choice = infect_choice(history,nb_conta_av,k)
        evolution = evolve(history,k)
        update = infect_update(evolution,choice,k)

        for x in range(xmax) :
            for y in range(ymax) :
                v = 0 #Healthy

                if update[x][y][1] == 1 and update[x][y][3] == 0 :
                    v = 1 #Asymptomac

                if update[x][y][1] == 1 and update[x][y][3] == 1 :
                    v = 2 #Minor

                if update[x][y][1] == 1 and update[x][y][3] == 2 :
                    v = 3 #Severe

                if update[x][y][1] == 1 and update[x][y][3] == 3 :
                    v = 4 #Dead

                if update[x][y][1] == 2 :
                    v = 5 #Healed

                history_anim[k][x][y] = v
        history = update
    return history_anim


#Select data we want for different graph
def history_death(initialization,time) :
    count = 0
    list = []
    history_anim = np.zeros((time,xmax,ymax))
    history = initialization

    for k in range(time) :
        choice = infect_choice(history,nb_conta_av,k)
        evolution = evolve(history,k)
        update = infect_update(evolution,choice,k)

        for x in range(xmax) :
            for y in range(ymax) :
                if update[x][y][1] == 1 and update[x][y][3] == 3 :
                    count += 1 #Dead

        history = update
        list.append(count)
        count = 0
    return [list,"Number of death"]


def history_infect(initialization,time) :
    count = 0
    list = []
    history_anim = np.zeros((time,xmax,ymax))
    history = initialization

    for k in range(time) :
        choice = infect_choice(history,nb_conta_av,k)
        evolution = evolve(history,k)
        update = infect_update(evolution,choice,k)

        for x in range(xmax) :
            for y in range(ymax) :
                if update[x][y][1] == 1 and update[x][y][3] != 3  :
                    count += 1 #Infected

        history = update
        list.append(count)
        count = 0
    return [list,"Number of infected"]



def history_healed(initialization,time) :
    count = 0
    list = []
    history_anim = np.zeros((time,xmax,ymax))
    history = initialization

    for k in range(time) :
        choice = infect_choice(history,nb_conta_av,k)
        evolution = evolve(history,k)
        update = infect_update(evolution,choice,k)

        for x in range(xmax) :
            for y in range(ymax) :
                if update[x][y][1] == 2 :
                    count += 1 #Healed

        history = update
        list.append(count)
        count = 0
    return [list,"Number of Healed"]

def history_infect_age0(initialization,time) :
    count = 0
    list = []
    history_anim = np.zeros((time,xmax,ymax))
    history = initialization

    for k in range(time) :
        choice = infect_choice(history,nb_conta_av,k)
        evolution = evolve(history,k)
        update = infect_update(evolution,choice,k)

        for x in range(xmax) :
            for y in range(ymax) :
                if update[x][y][0] == 0 and update[x][y][1] == 1 and update[x][y][3] != 3 :
                    count += 1 #Infected and age0

        history = update
        list.append(count)
        count = 0
    return [list,"Number of infected age0"]



def history_severe(initialization,time) :
    count = 0
    list = []
    history_anim = np.zeros((time,xmax,ymax))
    history = initialization

    for k in range(time) :
        choice = infect_choice(history,nb_conta_av,k)
        evolution = evolve(history,k)
        update = infect_update(evolution,choice,k)

        for x in range(xmax) :
            for y in range(ymax) :
                if update[x][y][2] == 2 and update[x][y][1] == 1 :
                    count += 1 #Severe

        history = update
        list.append(count)
        count = 0
    return [list,"Number of severe cases"]

def graph(history) :
    y = history[0]
    x = np.arange(0,len(y),1)
    plt.plot(x,y,label = history[1], color='limegreen')
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("Days")
    plt.ylabel("Cases")
    plt.show()


def makeMovie(history):


    FIGSIZE = (16,9)
    DPI = 240


    norm=plt.Normalize(0,5)
    my_cmap = mcolors.ListedColormap(["forestgreen", "yellow", "darkorange", "red", "black","cyan"])
    fig = plt.figure(figsize=FIGSIZE,dpi=DPI)
    ax = fig.add_subplot(111)

    im  = ax.imshow(history[0,:,::-1].T, cmap=my_cmap, vmin = 0, vmax = 5)


    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()

    cnt = ax.text(0.01, 0.99, str(0),color='white', fontsize=15,
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes)
    mesure = ax.text(0.99, 0.99, str(0),color='white', fontsize=15,
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes)

    def update_img(n):
        im.set_data(history[n,:,::-1].T)
        cnt.set_text(str(n))
        if day_measures_func(n)[0] == 1 :
            mesure.set_text("Closing schools")
        if day_measures_func(n)[1] == 1 :
            mesure.set_text("Confinement")
        if day_measures_func(n)[2] == 1 :
            mesure.set_text("Quarantine")
        if day_measures_func(n)[3] == 1 :
            mesure.set_text("Barrier gestures")
        if day_measures_func(n)[4] == 1 :
            mesure.set_text("Isolation age")
        if day_measures_func(n)[5] == 1 :
            mesure.set_text("Isolation Infected")
        if day_measures_func(n)[0] == 0 and day_measures_func(n)[1] == 0 and day_measures_func(n)[2] == 0 and day_measures_func(n)[3] == 0 and day_measures_func(n)[4] == 0 and day_measures_func(n)[5] == 0:
            mesure.set_text("")
        return True


    ani = FuncAnimation(fig, update_img, history.shape[0],repeat=False, interval=5)
    plt.show()


















