# Tapid19

Code made during EuVsVirus Hackathon.

The user interface is not implemented so to use it you have to setup the settings and follow these instructions :
(Some implemented fonctionalities are not optimized without user interface but you still can use it by reading the code)

FIRST : open the program in your python IDE , copy the pragraph #settings from this text document right after #settings in the code.

SECOND : set up your own settings with the help of the following explainations :

time  : it is the duration of your simulation
         modify k to setup your duration (more than 200 is generally useless)
         
population : it is the number of person in your population
              modify p to setup your population (you will need a powerfull computer for more than 100,000 people)

R0 : it is the number of people that one infected contaminates
      modify r to setup your R0 (Covid19 R0 is between 2 and 3)
      
nb_infected : it is the number of infected that you want at the beginning of the simulation
              modify i to setup your number of infected

prop_ages : it is the proportion of every class of age (young, adult, old)
              it is already completed if you want classic values or modify it to setup your own (the total has to be equal to 1)

nb_meet : it is a table representing the number people that every person of a class of age (young, adult, old) meet in one day 
          it is already completed if you want classic or modify it to setup your own values

coord_infected : OPTIONAL fonctionality
                  it is the coordonates of your infected at the beginning (this fonctionality is not optimized until the user interface 
                   is not implemented because you could pick out of bounds coordonates)
                  if you want to select these coordonates (for example to match with quarantine measure) you can modify by adding 
                  coordonates like that : coord_infected = [(x1,y1),(x2,y2),...]

scenario : OPTIONAL fonctionality
            it is the tabe of the different measures you want to take with the date of begining and ending for every measures 
           (again this is not optimized until the user interface is not implemented because it is hard to modify it with formalism needed)
              if you want to select the measure please modify scenario like that :
              [[measures_list[k1][0],begin_date1,ending_date1],[measures_list[k2][0],begin_date2,ending_date2],...]
              k1 (or k2) represent the measure : 
                  -k1 = 0 : closing schools
                  -k1= 1: confinement
                  -k1 = 2 : quarantine (not optimized without user interface because you can not select the area of quarantine)
                  -k1 = 3 : barrier gestures
                  -k1 = 4 : isolation for a class of age (not optimized without user interface because you can not select the age)
                  -k1 = 5 : isolation of infected

THIRD : To run the program you have to copy in the console these different command depending of what you want :
         
        - animation : makeMovie(history_ani(initialization_table(),time))
        - graph death : graph(history_death(initialization_table(),time))
        - graph infected : graph(history_infect(initialization_table(),time))
        - graph healed : graph(history_healed(initialization_table(),time))
        - graph infected young : graph(history_infect_age0(initialization_table(),time))
        - graph severe cases : graph(history_severe(initialization_table(),time))
        
        MORE graph will be added soon


################################################################################################


         #settings

         time = t
         population = p
         R0 = r
         nb_infected = i
         prop_ages = [0.2,0.6,0.2]
         nb_meet = np.array([[50,10,5],[10,50,10],[5,10,20]])
         coord_infected = []
         scenario = []
